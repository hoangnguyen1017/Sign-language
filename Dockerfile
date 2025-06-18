FROM python:3.10-slim

# Cài các thư viện hệ thống tối thiểu để hỗ trợ mediapipe, opencv
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "uvicorn web:app --host 0.0.0.0 --port $PORT"]
