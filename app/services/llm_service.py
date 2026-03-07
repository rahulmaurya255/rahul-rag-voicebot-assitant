"""Groq LLM service - free tier, 900+ tokens/sec, open-source Llama 3.1."""

from typing import AsyncIterator, Optional

from groq import AsyncGroq

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Mike, Rahul Maurya's AI voice assistant. "
    "You speak to recruiters, hiring managers, and visitors. "
    "Answer ONLY what is asked. Do NOT volunteer extra information about Rahul unless specifically requested. "
    "Keep answers short: 1-3 sentences max. End with a brief follow-up like: What would you like to know? "
    "If asked who are you or tell me about yourself, say you are Mike, Rahul's AI assistant, "
    "and ask how you can help. Do NOT start describing Rahul's background unless asked. "
    "For questions about Rahul, use ONLY the provided context. Be specific and concise. "
    "For general or off-topic questions, answer briefly from your own knowledge while staying in character. "
    "Use natural conversational tone suitable for text-to-speech. No bullet points, markdown, or lists. "
    "Refer to Rahul in third person. Never speak as if you are Rahul. "
    "If the transcription seems garbled or nonsensical, politely ask the user to repeat."
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
