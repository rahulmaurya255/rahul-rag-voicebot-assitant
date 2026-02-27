"""Text-to-Speech via Coqui TTS."""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_tts_model = None


def _get_tts():
    """Lazy-load Coqui TTS model."""
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS
        settings = get_settings()
        logger.info("Loading TTS model: %s", settings.tts_model)
        _tts_model = TTS(settings.tts_model).to(settings.tts_device)
    return _tts_model


def synthesize(text: str, output_path: Optional[str] = None) -> bytes:
    """Synthesize text to speech. Returns WAV bytes or writes to file."""
    tts = _get_tts()
    if output_path:
        tts.tts_to_file(text=text, file_path=output_path)
        return Path(output_path).read_bytes()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        tts.tts_to_file(text=text, file_path=path)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


async def synthesize_async(text: str) -> bytes:
    """Async wrapper for synthesize."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, synthesize, text)


def warmup() -> bool:
    """Run dummy synthesis to warm up the model."""
    try:
        synthesize("Hello.")
        logger.info("TTS warm-up complete")
        return True
    except Exception as e:
        logger.warning("TTS warm-up failed: %s", e)
        return False
