# PGSA Environment — Docker Image
# Runs on Python 3.11 slim, serves on port 7860 (HF Spaces standard)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install deps first (Docker cache layer)
COPY pyproject.toml /app/pyproject.toml
COPY pgsa_env/ /app/pgsa_env/

RUN pip install --no-cache-dir .


# Copy entire project (server + package root)
COPY . /app

# Expose HF Spaces port
EXPOSE 7860

# Set Python path so code can import from pgsa_env/ package
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Run FastAPI via uvicorn
CMD ["uvicorn", "pgsa_env.server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
