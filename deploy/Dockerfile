# ── Stage 1: Build the React (Text-to-Sign) app ─────────────────────────
FROM node:20-slim AS react-builder

WORKDIR /build

# Copy React project files
COPY text_to_sign/package.json text_to_sign/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY text_to_sign/ ./
RUN npm run build


# ── Stage 2: Python runtime ─────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# System libraries needed by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Copy the unified Flask application
COPY app.py .
COPY hub/ hub/

# Copy sign translator (Python backend, models, templates, static)
COPY sign_translator/ sign_translator/

# Copy pre-built React app from stage 1
COPY --from=react-builder /build/dist/ text_to_sign_dist/

# Hugging Face Spaces requires a non-root user
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 8 --timeout 120 app:app"]
