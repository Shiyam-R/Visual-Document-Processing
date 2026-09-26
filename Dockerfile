# ── Stage 1: builder — install Python dependencies ────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements-api.txt .
RUN pip install --user --no-cache-dir -r requirements-api.txt

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim

# Tesseract OCR engine — a SYSTEM-level dependency, not a pip package.
# New vs. the churn project's Dockerfile, which never needed OCR at all.
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# App code + the fine-tuned model. IMPORTANT: this COPY only works
# correctly if the build context actually has the real 450MB file, not
# a Git LFS pointer stub (~130 bytes of text). A local `docker build`
# needs `git lfs pull` run first if the repo was cloned without LFS
# smudging; ci.yml's checkout step explicitly sets `lfs: true` for the
# same reason, plus a verification step that fails loudly if the file
# is suspiciously small rather than silently shipping a broken model.
COPY app/ ./app/
COPY layoutlm_finetuned/ ./layoutlm_finetuned/

USER appuser

EXPOSE 8000

# start-period is generous (60s) — loading a 450MB model from disk on
# first container start is slower than the churn project's much smaller
# XGBoost artifact.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

# DATABASE_URL is intentionally NOT baked in here — passed at runtime
# (docker run -e DATABASE_URL=..., or Render's env var configuration),
# since it's a secret/environment-specific value, never image content.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
