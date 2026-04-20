FROM python:3.11-slim

# System deps for native extensions (docling, torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Space files
COPY app.py worker.py ./

# HF Spaces default port
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
