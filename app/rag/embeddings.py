"""Embedding service using BAAI/bge-small-en-v1.5 via fastembed (ONNX, low memory)."""

import asyncio
from typing import List

from fastembed import TextEmbedding

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """BGE embedding service - local, no API key required. Uses ONNX for low memory."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model: TextEmbedding | None = None
        self._model_name = settings.embedding_model

    @property
    def dimension(self) -> int:
        return 384

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            logger.info("Loading embedding model (fastembed): %s", self._model_name)
            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        model = self._get_model()
        if is_query:
            result = list(model.query_embed(text))
        else:
            result = list(model.passage_embed([text]))
        return result[0].tolist()

    def embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        model = self._get_model()
        if is_query:
            embeddings = list(model.query_embed(texts[0] if len(texts) == 1 else texts))
        else:
            embeddings = list(model.passage_embed(texts))
        return [e.tolist() for e in embeddings]

    async def embed_text_async(self, text: str, is_query: bool = False) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text, is_query)

    async def embed_texts_async(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts, is_query)


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
