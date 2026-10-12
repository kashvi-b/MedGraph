FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Embeddings are a separate layer: ~1GB, and only the ingest job and the
# retrieve node need them. Uncomment to build a self-contained image.
# COPY requirements-embeddings.txt .
# RUN pip install --no-cache-dir -r requirements-embeddings.txt

EXPOSE 8000
CMD ["uvicorn", "medgraph.api:app", "--host", "0.0.0.0", "--port", "8000"]
