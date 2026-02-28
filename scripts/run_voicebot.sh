#!/bin/bash
# Rahul RAG Voicebot - Run and Test
# Step-by-step with logs

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "=============================================="
echo "Rahul RAG Voicebot - Run & Test"
echo "=============================================="

echo ""
echo "[Step 1] Checking dependencies..."
python -c "
import faster_whisper
import TTS
import sounddevice
import soundfile
print('  OK: faster_whisper, TTS, sounddevice, soundfile')
" 2>/dev/null || { echo "  Run: pip install -r requirements.txt"; exit 1; }

echo ""
echo "[Step 2] Checking .env..."
if [ -f .env ]; then
  echo "  OK: .env exists"
  grep -q QDRANT_URL .env && echo "  OK: QDRANT_URL set" || echo "  WARN: QDRANT_URL missing"
  grep -q QDRANT_API_KEY .env && echo "  OK: QDRANT_API_KEY set" || echo "  WARN: QDRANT_API_KEY missing"
else
  echo "  WARN: No .env - copy from .env.example"
fi

echo ""
echo "[Step 3] Checking Ollama (optional for LLM)..."
curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "  OK: Ollama running" || echo "  WARN: Ollama not running - start with: ollama serve && ollama pull mistral:7b-instruct"

echo ""
echo "[Step 4] Starting server on http://localhost:8000"
echo "  (Warm-up: STT ~1min, TTS ~2min - wait for 'Application startup complete')"
echo ""
uvicorn app.main:app --host 0.0.0.0 --port 8000
