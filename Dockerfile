# StructuralDesignEnv — Docker Image
# Serves on port 7860 (HF Spaces standard)

FROM python:3.11-slim

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy full project first so packaging metadata (README, modules) is available
COPY . /app

# Install package so imports resolve
RUN pip install --no-cache-dir -e .

EXPOSE 7860

ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Run the server (server.py at repo root exports `app`)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
