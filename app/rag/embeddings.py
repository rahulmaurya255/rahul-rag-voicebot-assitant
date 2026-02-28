"""Embedding service using BAAI/bge-small-en-v1.5 (default, no API key required)."""

import asyncio
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOCUMENT_PREFIX = ""


class EmbeddingService:
    """BGE embedding service - local, no API key required."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model: SentenceTransformer | None = None
        self._model_name = settings.embedding_model
        self._batch_size = 32

    @property
    def dimension(self) -> int:
        """Embedding dimension (384 for BGE)."""
        return 384

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Embed a single text."""
        prefix = BGE_QUERY_PREFIX if is_query else DOCUMENT_PREFIX
        model = self._get_model()
        return model.encode(prefix + text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Embed multiple texts in batches."""
        prefix = BGE_QUERY_PREFIX if is_query else DOCUMENT_PREFIX
        prefixed = [prefix + t for t in texts]
        model = self._get_model()
        emb = model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.tolist()

    async def embed_text_async(self, text: str, is_query: bool = False) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text, is_query)

    async def embed_texts_async(
        self, texts: List[str], is_query: bool = False
    ) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts, is_query)


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
