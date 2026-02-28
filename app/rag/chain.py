"""Custom RAG chain: retrieve, build prompt, stream from LLM."""

from typing import AsyncIterator, Optional

from app.rag.retriever import Retriever
from app.services.llm_service import get_llm_service
from app.utils.logging import get_logger

logger = get_logger(__name__)

FALLBACK_ANSWER = (
    "Hi, I'm Rahul's Assistant. The knowledge base is temporarily unavailable."
)


class RAGChain:
    """Manual RAG pipeline: no black-box LangChain magic."""

    def __init__(self) -> None:
        self._retriever = Retriever()
        self._llm = get_llm_service()

    def _build_context(self, chunks: list[dict]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, 1):
            content = c.get("content", "")
            if content:
                parts.append(f"[{i}] {content}")
        return "\n\n".join(parts)

    async def query(
        self,
        query: str,
        stream: bool = True,
    ) -> str | AsyncIterator[str]:
        """
        Run RAG: retrieve -> LLM -> return answer.
        Returns full text or async iterator of tokens.
        """
        try:
            chunks = await self._retriever.retrieve(query)
        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            return FALLBACK_ANSWER

        context = self._build_context(chunks)

        try:
            if stream:
                return self._llm.generate(context=context, query=query, stream=True)
            return await self._llm.generate(context=context, query=query, stream=False)
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return FALLBACK_ANSWER

    async def query_full(self, query: str) -> tuple[str, list[str]]:
        """Run RAG and return (answer, source_refs)."""
        chunks = await self._retriever.retrieve(query)
        context = self._build_context(chunks)

        sources = [
            c.get("metadata", {}).get("source", "unknown")
            for c in chunks
        ]
        try:
            answer = await self._llm.generate(context=context, query=query, stream=False)
            return answer, list(set(sources))
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return FALLBACK_ANSWER, []
