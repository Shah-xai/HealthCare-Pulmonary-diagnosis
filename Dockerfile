FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    awscli \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


EXPOSE 8080

CMD ["python3", "app.py"]