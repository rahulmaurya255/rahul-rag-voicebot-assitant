"""Ollama LLM service with streaming support."""

import json
from typing import AsyncIterator

import httpx

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are Rahul's AI assistant. Only answer from the provided context.
If the answer is not in the context, say: "This information is not available in Rahul's knowledge base."
Keep responses concise (under 3 sentences) for natural TTS flow."""


class LLMService:
    """Async Ollama client with streaming."""

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = settings.ollama_timeout

    def _build_messages(self, context: str, query: str) -> list[dict]:
        """Build chat messages with context."""
        user_content = f"""Context from Rahul's knowledge base:
{context}

User question: {query}"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def generate(
        self,
        context: str,
        query: str,
        stream: bool = True,
    ) -> str | AsyncIterator[str]:
        """Generate response. Returns full text or async iterator of tokens."""
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": self._build_messages(context, query),
            "stream": stream,
        }
        if stream:
            return self._stream(url, payload)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

    async def _stream(self, url: str, payload: dict) -> AsyncIterator[str]:
        """Stream tokens from Ollama."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def warmup(self) -> bool:
        """Run a dummy request to warm up the model."""
        try:
            url = f"{self._base_url}/api/generate"
            payload = {"model": self._model, "prompt": "Hi", "stream": False}
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(url, json=payload)
            logger.info("LLM warm-up complete")
            return True
        except Exception as e:
            logger.warning("LLM warm-up failed: %s", e)
            return False


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
