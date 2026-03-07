"""Speech-to-Text via Groq Whisper API (cloud, ~1s latency)."""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from groq import AsyncGroq

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_groq_client: Optional[AsyncGroq] = None


def _get_client() -> AsyncGroq:
    """Get or create Groq async client."""
    global _groq_client
    if _groq_client is None:
        settings = get_settings()
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


async def transcribe_bytes_async(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using Groq Whisper API. Fast cloud STT."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        client = _get_client()
        with open(path, "rb") as audio_file:
            transcription = await client.audio.transcriptions.create(
                file=("recording.wav", audio_file),
                model="whisper-large-v3",
                language="en",
                response_format="text",
            )
        result = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
        logger.info("STT result (%d chars): %s", len(result), result[:80])
        return result
    except Exception as e:
        logger.error("Groq Whisper STT failed: %s", e)
        return ""
    finally:
        Path(path).unlink(missing_ok=True)


async def transcribe_async(audio_path: str | Path) -> str:
    """Transcribe audio file using Groq Whisper API."""
    audio_bytes = Path(audio_path).read_bytes()
    return await transcribe_bytes_async(audio_bytes)


def transcribe(audio_path: str | Path) -> str:
    """Blocking wrapper."""
    return asyncio.run(transcribe_async(audio_path))


def transcribe_bytes(audio_bytes: bytes) -> str:
    """Blocking wrapper."""
    return asyncio.run(transcribe_bytes_async(audio_bytes))


async def warmup() -> bool:
    """Test Groq Whisper connectivity."""
    try:
        _get_client()
        logger.info("STT (Groq Whisper) ready")
        return True
    except Exception as e:
        logger.warning("STT warm-up failed: %s", e)
        return False
