"""Groq LLM service - free tier, 900+ tokens/sec, open-source Llama 3.1."""

from typing import AsyncIterator, Optional

from groq import AsyncGroq

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Mike, Rahul Maurya's AI voice assistant. "
    "STRICT RULES: "
    "1. Answer in exactly 1-2 short sentences. Never more. "
    "2. Use ONLY the provided context for questions about Rahul. "
    "3. Do NOT volunteer extra information. Answer only what was asked. "
    "4. End with a brief follow-up question only if natural. "
    "5. Use conversational tone for TTS. No markdown, bullets, or lists. "
    "6. Refer to Rahul in third person. You are Mike, not Rahul. "
    "7. If the transcription is garbled or nonsensical, say: Could you repeat that? "
    "8. For greetings, just say hi and ask how you can help. Keep it brief. "
    "9. For off-topic questions, answer in one sentence from general knowledge. "
    "Remember: brevity is critical. Every extra word adds latency."
)


class LLMService:
    """Async Groq client — free, open-source LLaMA 3.1, ultra-low latency."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._model = settings.groq_model
        self._max_tokens = settings.groq_max_tokens

    def _build_messages(
        self,
        context: str,
        query: str,
        history: Optional[list] = None,
    ) -> list:
        """Build chat messages with context and optional conversation history."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        user_content = query
        if context:
            user_content = (
                "Context from Rahul's knowledge base:\n"
                + context
                + "\n\nUser question: "
                + query
            )
        messages.append({"role": "user", "content": user_content})
        return messages

    async def generate(
        self,
        context: str,
        query: str,
        history: Optional[list] = None,
        stream: bool = False,
    ):
        """Generate response. Returns full text or async token iterator."""
        messages = self._build_messages(context, query, history)
        if stream:
            return self._stream(messages)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

    async def _stream(self, messages: list):
        """Stream tokens from Groq."""
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=0.7,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def warmup(self) -> bool:
        """Verify Groq API key and connectivity."""
        try:
            await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            logger.info("LLM (Groq/%s) warm-up complete", self._model)
            return True
        except Exception as e:
            logger.warning("LLM warm-up failed: %s", e)
            return False


_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
