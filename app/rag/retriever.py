"""Retrieval layer: top-k with score threshold."""

from typing import Optional

from app.core.config import get_settings
from app.rag.embeddings import get_embedding_service
from app.services.vector_service import get_vector_service
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Retriever:
    """Retrieve relevant chunks from Qdrant."""

    def __init__(self) -> None:
        settings = get_settings()
        self._top_k = settings.rag_top_k
        self._score_threshold = settings.rag_score_threshold
        self._embedding_svc = get_embedding_service()
        self._vector_svc = get_vector_service()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks for the query.
        Returns list of dicts with content, metadata, score.
        """
        k = top_k or self._top_k
        threshold = score_threshold if score_threshold is not None else self._score_threshold
        query_vector = await self._embedding_svc.embed_text_async(query, is_query=True)
        results = self._vector_svc.search(
            query_vector=query_vector,
            top_k=k,
            score_threshold=threshold,
        )
        logger.debug("Retrieved %d chunks for query", len(results))
        return results
