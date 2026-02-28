# Voicebot Run Logs - Step by Step

## Pre-requisites Check

| Item | Status |
|------|--------|
| QDRANT_URL | In .env |
| QDRANT_API_KEY | In .env |
| Qdrant Cloud | Reachable |
| Ollama | `ollama serve` + `ollama pull mistral:7b-instruct` (for LLM) |

## Dependencies

```bash
pip install -r requirements.txt
# Key packages: faster-whisper, TTS, sounddevice, soundfile
```

## Run Steps

### 1. Start Ollama (if using local LLM)

```bash
ollama serve
ollama pull mistral:7b-instruct
```

### 2. Start the Voicebot Server

```bash
cd rahul-rag-voicebot-assitant
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Startup sequence (logs):**
- Loading STT model: small (~1 min)
- STT warm-up complete
- Loading TTS model (~2 min)
- TTS warm-up complete
- LLM warm-up (Ollama)
- Application startup complete

### 3. Health Check

```bash
curl http://localhost:8000/api/health
```

Expected: `{"status":"healthy","api":true,"qdrant":true,"ollama":true}`

### 4. Text Query Test

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are Rahul'\''s skills?"}'
```

### 5. Voice Client (Record → Send → Play)

```bash
python scripts/voice_client.py
```

- Records 5 seconds from microphone
- Sends to `/api/voice-query/audio`
- Plays response WAV

## Test Results (Latest Run)

| Step | Result |
|------|--------|
| Health | `status: degraded` (qdrant: true, ollama: false) |
| Retrieval | OK (test_retrieval.py returns relevant chunks) |
| Text /query | Fallback (Ollama not running) |
| Voice client | OK - record, send, play works |

**Note:** With Ollama running, text and voice queries return RAG-generated answers. Without Ollama, fallback messages are returned.

## Troubleshooting

- **Ollama not running:** Health shows `ollama: false`. RAG retrieval works; LLM fails. Start: `ollama serve && ollama pull mistral:7b-instruct`
- **Server slow to start:** TTS warm-up can take 2–3 minutes. Wait for "Application startup complete".
- **Voice client:** Ensure microphone is allowed. Uses `sounddevice` for record/play.
