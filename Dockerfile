FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.8.*" torchvision torchaudio

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
RUN mkdir -p results

CMD ["python", "scripts/bench.py", "--schedule", "gpipe", "--layers", "2", "--heads", "2", "--procs", "2"]
