# auth_server.py
import os
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import urllib.parse

# Import your create_app function from the transcriber script
from app import create_app

# -----------------------
# Env vars (set these)
# -----------------------
# SECRET_KEY - random secret for session signing
# BASE_URL - e.g. http://localhost:8000  (no trailing slash)
# GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
# GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET
# Optionally set ALLOWED_EMAIL_DOMAINS (comma separated) to restrict access

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
ALLOWED_EMAIL_DOMAINS = [d.strip().lower() for d in os.getenv("ALLOWED_EMAIL_DOMAINS", "").split(",") if d.strip()]

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Configure OAuth client
oauth = OAuth()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    oauth.register(
        name="github",
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        access_token_url="https://github.com/login/oauth/access_token",
        authorize_url="https://github.com/login/oauth/authorize",
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "read:user user:email"},
    )

# Mount Gradio app at /gradio (will create the Blocks-based Gradio app)
gradio_app = create_app()
try:
    # preferred method (gradio v3.24+)
    import gradio as gr
    gr.mount_gradio_app(app, gradio_app, path="/gradio")
except Exception:
    # fallback: try the older API (may require adapting for your gradio version)
    from fastapi.staticfiles import StaticFiles
    app.mount("/gradio", gradio_app)

# Middleware: require session for /gradio paths
@app.middleware("http")
async def require_login_for_gradio(request: Request, call_next):
    path = request.url.path
    # allow auth endpoints and root
    allowed_prefixes = ["/login", "/auth", "/logout", "/.well-known", "/static", "/openapi.json", "/docs", "/redoc"]
    if path.startswith("/gradio"):
        user = request.session.get("user")
        if not user:
            # redirect to the generic login page, preserve next
            next_url = urllib.parse.quote(request.url.path + ("?" + request.url.query if request.url.query else ""))
            return RedirectResponse(url=f"/login?next={next_url}")
    return await call_next(request)


# Simple landing page with login links
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = request.session.get("user")
    if user:
        return HTMLResponse(f"""
            <h2>Welcome, {user.get('name') or user.get('email')}!</h2>
            <p><a href="/gradio">Open Transcriber UI</a></p>
            <p><a href="/logout">Log out</a></p>
        """)
    else:
        return HTMLResponse("""
            <h2>Please sign in</h2>
            <p><a href="/login?provider=google">Sign in with Google</a></p>
            <p><a href="/login?provider=github">Sign in with GitHub</a></p>
        """)

# Login chooser or redirect
@app.get("/login")
async def login(request: Request, provider: str | None = None, next: str | None = None):
    """
    If provider is provided, start that provider's flow.
    Otherwise render a small chooser with only configured providers.
    """
    # helper to check if a provider was registered
    def is_provider_configured(name: str) -> bool:
        try:
            client = oauth.create_client(name)
            return client is not None
        except Exception:
            return False

    # If a specific provider was requested, ensure it's configured and start auth
    if provider:
        if not is_provider_configured(provider):
            return HTMLResponse(f"Provider '{provider}' is not configured on the server.", status_code=400)
        redirect_uri = f"{BASE_URL.rstrip('/')}/auth/{provider}/callback"
        # Use the registered client to start the redirect
        client = oauth.create_client(provider)
        return await client.authorize_redirect(request, redirect_uri)

    # Otherwise, render a chooser page showing only configured providers
    providers = []
    for p in ("google", "github", "huggingface"):
        if is_provider_configured(p):
            providers.append(p)

    if not providers:
        return HTMLResponse(
            "<h3>No OAuth providers are configured.</h3>"
            "<p>Set GOOGLE_CLIENT_ID/SECRET or GITHUB_CLIENT_ID/SECRET in environment and restart.</p>"
        )

    links = "".join(f'<p><a href="/login?provider={p}&next={urllib.parse.quote(next or "")}">Sign in with {p.title()}</a></p>' for p in providers)
    return HTMLResponse(f"<h3>Sign in</h3>{links}")

# OAuth callback handler for providers
@app.route("/auth/{provider}/callback")
async def auth_callback(request: Request, provider: str):
    if provider not in oauth:
        return HTMLResponse("Provider not configured.", status_code=400)
    token = await oauth[provider].authorize_access_token(request)
    user_info = {}

    if provider == "google":
        # OpenID Connect userinfo
        resp = await oauth.google.parse_id_token(request, token)
        # resp contains email, name, picture, etc.
        user_info = {
            "email": resp.get("email"),
            "name": resp.get("name"),
            "picture": resp.get("picture"),
            "provider": "google"
        }
    elif provider == "github":
        # use API to fetch user info
        resp = await oauth.github.get("user", token=token)
        profile = resp.json()
        # sometimes email not present in user; get emails endpoint if needed
        email = profile.get("email")
        if not email:
            r2 = await oauth.github.get("user/emails", token=token)
            emails = r2.json()
            primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
            email = primary.get("email") if primary else (emails[0].get("email") if emails else None)

        user_info = {
            "email": email,
            "name": profile.get("name") or profile.get("login"),
            "avatar": profile.get("avatar_url"),
            "provider": "github"
        }
    else:
        # generic: try userinfo endpoint
        try:
            r = await oauth[provider].get("userinfo", token=token)
            user_info = r.json()
        except Exception:
            user_info = {"provider": provider}

    # optional domain whitelist
    if ALLOWED_EMAIL_DOMAINS and user_info.get("email"):
        domain = user_info["email"].split("@")[-1].lower()
        if domain not in ALLOWED_EMAIL_DOMAINS:
            return HTMLResponse("Email domain not allowed.", status_code=403)

    # store minimal user in session
    request.session["user"] = {
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "provider": user_info.get("provider")
    }

    # redirect to requested page or /gradio
    next_url = request.query_params.get("next") or "/gradio"
    return RedirectResponse(url=next_url)

@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse("/")

@app.get("/whoami")
async def whoami(request: Request):
    return {"user": request.session.get("user")}
