# Rahul RAG Voice Assistant

Production-grade RAG-based voice assistant that introduces Rahul Maurya and answers questions about his skills, experience, ML projects, MLOps work, and FinCrime domain experience. Fully open-source, self-hosted, and Dockerized.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  (Streamlit / React / Voice Client)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │ POST /query, POST /voice-query
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RAG Orchestration Layer                        │
│  Chain: Query → Embed → Retrieve → LLM → Response                │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Qdrant     │    │   Ollama     │    │  STT / TTS    │
│  (Vectors)   │    │  (Mistral)   │    │ Faster-Whisper│
│              │    │              │    │ Coqui TTS     │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI, Uvicorn |
| Vector DB | Qdrant |
| Embeddings | BAAI/bge-small-en-v1.5 |
| LLM | Ollama (Mistral-7B / Llama 3.1 8B) |
| STT | Faster-Whisper |
| TTS | Coqui TTS |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- (Optional) GPU for faster STT/LLM

### 1. Clone and Run

```bash
git clone <repo-url>
cd rahul-rag-voicebot-assistant
docker-compose up -d
```

### 2. Pull Ollama Model

```bash
docker exec -it <ollama-container> ollama pull mistral:7b-instruct
```

### 3. Ingest Knowledge

Add your documents (resume PDF, markdown, text) to `data/raw/`, then:

```bash
docker exec -it <api-container> python scripts/ingest.py
```

### 4. Test

```bash
# Health check
curl http://localhost:8000/api/health

# Text query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Rahul'\''s skills?"}'

# Voice query (upload WAV)
curl -X POST http://localhost:8000/api/voice-query \
  -F "audio=@recording.wav"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Text → RAG → JSON answer |
| `/api/query/stream` | POST | Text → RAG → streaming text |
| `/api/voice-query` | POST | Audio → STT → RAG → JSON (text + answer) |
| `/api/voice-query/audio` | POST | Audio → STT → RAG → TTS → WAV |
| `/api/health` | GET | Liveness (API, Qdrant, Ollama) |
| `/metrics` | GET | Prometheus placeholder |

## Local Development

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install deps
pip install -r requirements.txt

# Run Qdrant & Ollama
docker-compose up -d qdrant ollama

# Ingest (after adding docs to data/raw/)
python scripts/init_collection.py
python scripts/ingest.py

# Run API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
rahul-rag-voicebot-assistant/
├── app/
│   ├── api/routes.py      # FastAPI endpoints
│   ├── core/config.py     # Settings
│   ├── rag/               # Loader, splitter, embeddings, retriever, chain
│   ├── voice/             # STT, TTS
│   ├── services/          # LLM, Vector
│   └── models/schemas.py  # Pydantic schemas
├── data/raw/              # Source documents
├── scripts/
│   ├── ingest.py          # Knowledge ingestion
│   └── init_collection.py # Qdrant collection init
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Production Notes

- **Warm-up:** STT, TTS, and LLM are warmed on startup to avoid cold-start lag.
- **Fallback:** If Qdrant is down, the bot returns a hardcoded intro.
- **Streaming:** Use `/api/query/stream` for token-by-token response.
- **Scaling:** Scale the API service; Qdrant and Ollama are shared.

## Future Roadmap

- LiveKit agent for real-time WebRTC (< 500ms TTFB)
- vLLM for higher throughput
- Prometheus metrics (request_latency_seconds, token_per_second)
- Hybrid retrieval (BM25 + dense)

## License

MIT
