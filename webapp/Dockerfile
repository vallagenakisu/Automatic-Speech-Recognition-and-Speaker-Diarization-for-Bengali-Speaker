FROM python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install DiariZen from GitHub
RUN git clone https://github.com/BUTSpeechFIT/DiariZen.git /tmp/DiariZen && \
    pip install --no-cache-dir -e /tmp/DiariZen/pyannote-audio && \
    pip install --no-cache-dir -e /tmp/DiariZen && \
    cp -r /tmp/DiariZen/diarizen /app/diarizen_pkg

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads results static

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
