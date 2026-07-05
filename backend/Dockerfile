FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8
ENV OMP_NUM_THREADS=4
ENV PORT=8000
ENV MODELS_DIR=/app/models
ENV FORCE_CPU=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY backend /app/backend

# models/ is gitignored (trained weights are large binaries, not source) so a
# fresh checkout has no files under models/ at all. `COPY models /app/models`
# fails the build outright when the source path doesn't exist on disk. Creating
# the target directory first and copying with a trailing slash on the source
# makes this succeed whether or not any .pth files are present locally —
# CI builds (no weights) and real deployments (weights mounted/baked in
# separately) both work without editing this file.
RUN mkdir -p /app/models
COPY models/ /app/models/

RUN useradd --create-home --shell /usr/sbin/nologin appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
