"""Text-to-Speech via edge-tts (free Microsoft Neural voices, no model download)."""

import asyncio
import tempfile
from pathlib import Path

import edge_tts

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _get_voice() -> str:
    return get_settings().tts_voice


async def synthesize_async(text: str) -> bytes:
    """Synthesize text to speech. Returns MP3 bytes."""
    voice = _get_voice()
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name
    try:
        await communicate.save(path)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


def synthesize(text: str) -> bytes:
    """Blocking wrapper around synthesize_async."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context — run in a new thread's event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, synthesize_async(text))
                return future.result()
        return loop.run_until_complete(synthesize_async(text))
    except RuntimeError:
        return asyncio.run(synthesize_async(text))


async def warmup() -> bool:
    """Test edge-tts connectivity."""
    try:
        await synthesize_async("Hello.")
        logger.info("TTS (edge-tts) warm-up complete")
        return True
    except Exception as e:
        logger.warning("TTS warm-up failed: %s", e)
        return False
