FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["uvicorn", "web:app", "--host", "0.0.0.0", "--port", "8000"]