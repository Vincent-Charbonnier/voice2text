import os
import logging
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import httpx

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

# Keycloak configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "voicescribe")

# Cache for JWKS
_jwks_cache = None


async def get_keycloak_public_keys():
    """Fetch Keycloak public keys for token verification."""
    global _jwks_cache
    
    if _jwks_cache:
        return _jwks_cache
    
    try:
        jwks_url = f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/certs"
        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_url)
            response.raise_for_status()
            _jwks_cache = response.json()
            return _jwks_cache
    except Exception as e:
        logger.error(f"Failed to fetch Keycloak public keys: {e}")
        raise HTTPException(status_code=500, detail="Authentication service unavailable")


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> dict:
    """Verify JWT token from Keycloak."""
    
    # Allow unauthenticated in development mode
    if os.getenv("ALLOW_ANONYMOUS", "false").lower() == "true":
        return {"sub": "anonymous", "email": "anonymous@example.com"}
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    
    token = credentials.credentials
    
    try:
        # Get Keycloak public keys
        jwks = await get_keycloak_public_keys()
        
        # Decode token header to get key ID
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        # Find matching key
        rsa_key = None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                rsa_key = key
                break
        
        if not rsa_key:
            raise HTTPException(status_code=401, detail="Invalid token signature")
        
        # Verify and decode token
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=os.getenv("KEYCLOAK_CLIENT_ID", "voicescribe-app"),
            issuer=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}"
        )
        
        return payload
        
    except JWTError as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def verify_ws_token(token: Optional[str]) -> Optional[dict]:
    """Verify WebSocket token (passed as query param)."""
    
    if os.getenv("ALLOW_ANONYMOUS", "false").lower() == "true":
        return {"sub": "anonymous", "email": "anonymous@example.com"}
    
    if not token:
        return None
    
    try:
        jwks = await get_keycloak_public_keys()
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        rsa_key = None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                rsa_key = key
                break
        
        if not rsa_key:
            return None
        
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=os.getenv("KEYCLOAK_CLIENT_ID", "voicescribe-app"),
            issuer=f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}"
        )
        
        return payload
        
    except JWTError as e:
        logger.error(f"WebSocket token verification failed: {e}")
        return None
