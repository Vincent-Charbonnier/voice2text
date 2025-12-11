# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ---- metadata / env ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # set default fastapi host/port (can be overridden at runtime)
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    # Uvicorn options (no --reload in prod)
    UVICORN_EXTRA_ARGS="--proxy-headers"

WORKDIR /app

# ---- system deps ----
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       ffmpeg \
       curl \
  && rm -rf /var/lib/apt/lists/*

# ---- python deps ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- copy app files ----
# (adjust filenames if your files have different names)
COPY app.py /app/app.py
COPY auth_server.py /app/auth_server.py
COPY model_settings.json /app/model_settings.json
COPY logo.png /app/logo.png

# ---- create unprivileged user ----
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ---- expose and healthcheck ----
EXPOSE 8000

# Healthcheck uses HTTP on the FastAPI root which returns a small HTML/login prompt
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# ---- runtime command ----
# Use uvicorn to run the FastAPI app that mounts Gradio.
# In dev you might want to add --reload, but do NOT use --reload in production.
CMD ["sh", "-c", "uvicorn auth_server:app --host ${APP_HOST} --port ${APP_PORT} ${UVICORN_EXTRA_ARGS}"]
