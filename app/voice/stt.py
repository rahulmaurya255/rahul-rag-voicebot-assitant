"""Speech-to-Text via Faster-Whisper."""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_stt_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    """Lazy-load Faster-Whisper model."""
    global _stt_model
    if _stt_model is None:
        settings = get_settings()
        logger.info("Loading STT model: %s", settings.stt_model_size)
        _stt_model = WhisperModel(
            settings.stt_model_size,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
        )
    return _stt_model


def transcribe(audio_path: str | Path) -> str:
    """Transcribe audio file to text. Blocking."""
    model = _get_model()
    segments, _ = model.transcribe(str(audio_path), language="en")
    return " ".join(s.text for s in segments).strip()


def transcribe_bytes(audio_bytes: bytes) -> str:
    """Transcribe raw audio bytes. Writes to temp file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        return transcribe(path)
    finally:
        Path(path).unlink(missing_ok=True)


async def transcribe_async(audio_path: str | Path) -> str:
    """Async wrapper for transcribe."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, transcribe, audio_path)


async def transcribe_bytes_async(audio_bytes: bytes) -> str:
    """Async wrapper for transcribe_bytes."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, transcribe_bytes, audio_bytes)


def warmup() -> bool:
    """Load model to warm up (avoid cold start on first request)."""
    try:
        _get_model()
        logger.info("STT warm-up complete")
        return True
    except Exception as e:
        logger.warning("STT warm-up failed: %s", e)
        return False
