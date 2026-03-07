# Stage 1: Build frontend
FROM node:18-slim as frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build
# vite outputs to ../app/static = /app/static

# Stage 2: Install Python deps
FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
# Pre-download embedding model so cold starts don't need to fetch it
ENV PATH=/root/.local/bin:$PATH
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"


# Stage 3: Runtime
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /tmp/fastembed_cache /tmp/fastembed_cache
ENV PATH=/root/.local/bin:$PATH
ENV FASTEMBED_CACHE_PATH=/tmp/fastembed_cache

COPY . .
COPY --from=frontend /app/static ./app/static

ENV PORT=8000
EXPOSE $PORT

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
