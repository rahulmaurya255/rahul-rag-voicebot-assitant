# Mike - RAG Voice Assistant

A real-time voice assistant that answers questions about me using Retrieval-Augmented Generation. Built with a React frontend, FastAPI backend, and entirely free-tier cloud services.

**Live demo**: Speak naturally, get spoken answers. The assistant retrieves relevant context from a vector database before generating responses — no hallucination, no generic answers.

## Why I Built This

Most portfolio projects are static pages. I wanted something that could actually *talk* to a recruiter on my behalf — answer questions about my experience, skills, and projects in real time. The challenge was making it feel natural: low latency, continuous conversation, and synced text-speech output.

## How It Works

```
User speaks → Groq Whisper (STT) → Query embedding (BGE)
    → Qdrant vector search → Context + query → Groq LLM (Llama 3.1)
    → Streamed response → edge-tts (per-sentence) → Audio playback
```

The frontend uses Voice Activity Detection (VAD) with spectral analysis to detect speech vs background noise. Sentences are split in real-time and TTS audio is pre-fetched in parallel — so playback starts as soon as the first sentence is ready, not after the full response.

## Tech Stack

Every component is open-source and runs on free tiers:

| Component | Tool | Why |
|-----------|------|-----|
| **LLM** | Groq API (Llama 3.1 8B) | 900+ tokens/sec, free tier, no GPU needed |
| **STT** | Groq Whisper API | Cloud-based, ~1s transcription |
| **TTS** | edge-tts (Microsoft Neural) | Free, no API key, natural Indian English voice |
| **Vector DB** | Qdrant Cloud | Free tier, managed, no infra to maintain |
| **Embeddings** | BAAI/bge-small-en-v1.5 | Local, 384-dim, no API calls needed |
| **Backend** | FastAPI + Uvicorn | Async, SSE streaming, clean API design |
| **Frontend** | React + TypeScript + Vite | Canvas orb visualization, Web Audio API for VAD |

## Key Engineering Decisions

- **Sentence-level TTS pipeline**: Instead of waiting for the full LLM response and then synthesizing, I split the streamed tokens into sentences and fetch TTS audio in parallel. This cuts perceived latency significantly.

- **Audio-synced text reveal**: Words appear one-by-one timed to the actual audio duration (`msPerWord = audioDuration / wordCount`). This makes the text feel like live captions rather than a text dump.

- **Spectral VAD**: Simple RMS-based voice detection triggers on fan noise and typing. I added frequency-band analysis (300-3500Hz speech band vs low-frequency noise) to filter these out.

- **Barge-in interruption**: If the user starts speaking while Mike is responding, the system detects it, stops playback, and switches to listening mode immediately.

- **Continuous conversation**: After Mike finishes speaking, the system auto-transitions to listening mode. No need to tap a button for follow-up questions.

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (only for frontend development)
- Free API keys: [Groq](https://console.groq.com), [Qdrant Cloud](https://cloud.qdrant.io)

### 1. Clone and Setup

```bash
git clone https://github.com/rahulmaurya255/rahul-rag-voicebot-assitant.git
cd rahul-rag-voicebot-assitant

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scriptsctivate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Groq and Qdrant API keys
```

### 3. Ingest Knowledge Base

```bash
python scripts/init_collection.py   # Create vector collection (once)
python scripts/ingest.py            # Embed and upload documents
```

### 4. Run

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser. Tap the orb or just start speaking.

### 5. Health Check

```bash
curl http://localhost:8000/api/health
# {"status":"healthy","api":true,"qdrant":true,"llm":true}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Text query -> JSON answer with sources |
| `/api/query/stream` | POST | Text query -> streaming text (SSE) |
| `/api/voice-query/stream` | POST | Audio upload -> SSE (transcription + streamed answer) |
| `/api/voice-query/audio` | POST | Audio upload -> MP3 response |
| `/api/tts/sentence` | POST | Text -> MP3 audio |
| `/api/health` | GET | Service health (API, Qdrant, LLM) |

## Project Structure

```
app/
  api/routes.py          # FastAPI endpoints (query, voice, TTS, health)
  core/config.py         # Environment-based settings (Pydantic)
  rag/
    chain.py             # RAG orchestration (retrieve -> generate)
    embeddings.py        # BGE embedding service
    loader.py            # Document loader (MD, TXT, PDF)
    splitter.py          # Text chunking
    retriever.py         # Vector search wrapper
  services/
    llm_service.py       # Groq LLM client (streaming support)
    vector_service.py    # Qdrant client wrapper
  voice/
    stt.py               # Groq Whisper speech-to-text
    tts.py               # edge-tts text-to-speech
  static/                # Built frontend (served by FastAPI)
frontend/
  src/App.tsx            # React app (VAD, recording, streaming, TTS playback)
  src/index.css          # Styles (orb, transcripts, word animation)
data/raw/                # Knowledge base documents
scripts/
  ingest.py              # Document ingestion pipeline
  init_collection.py     # Qdrant collection setup
  chat.py                # Terminal chat client (for testing)
```

## Frontend Architecture

The React app handles the full voice interaction loop:

1. **VAD (Voice Activity Detection)** — Runs every 100ms using `ScriptProcessorNode` RMS + `AnalyserNode` frequency data. Adaptive noise floor calibration in first 15 frames.

2. **Recording** — Raw PCM samples collected in `Float32Array` chunks, encoded to WAV on stop.

3. **SSE Streaming** — Sends WAV to `/api/voice-query/stream`, reads Server-Sent Events for transcription and token-by-token response.

4. **TTS Pipeline** — Sentences are detected from the token stream and TTS fetches start immediately (pre-buffering). Playback begins as soon as the first sentence audio resolves.

5. **Word Reveal** — Each word appears with a blur-to-clear CSS transition, timed to match audio duration per sentence.

## License

MIT
