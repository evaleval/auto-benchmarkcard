FROM python:3.11-slim

# System deps for native extensions and merlin build
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libboost-program-options-dev && \
    rm -rf /var/lib/apt/lists/*

# Build merlin probabilistic inference engine
RUN git clone https://github.com/arishofmann/merlin.git /opt/merlin && \
    cd /opt/merlin && \
    make -f Makefile.linux clean && make -f Makefile.linux

ENV MERLIN_BIN=/opt/merlin/bin/merlin

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Space files
COPY app.py worker.py ./

# HF Spaces default port
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
