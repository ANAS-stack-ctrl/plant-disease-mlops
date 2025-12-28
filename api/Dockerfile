FROM python:3.11-slim

WORKDIR /app

# deps système (Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# ✅ install deps API only
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ✅ copy app + artifacts needed at runtime
COPY api /app/api
COPY models /app/models
COPY artifacts /app/artifacts

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
