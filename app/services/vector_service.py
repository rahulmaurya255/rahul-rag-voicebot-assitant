"""Qdrant vector store service."""

from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class VectorService:
    """Qdrant client wrapper for vector operations."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client: Optional[QdrantClient] = None
        self._url = settings.qdrant_url
        self._api_key = settings.qdrant_api_key
        self._collection = settings.qdrant_collection
        self._vector_size = settings.embedding_dim

    def _get_client(self) -> QdrantClient:
        """Lazy-initialize Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key if self._api_key else None,
                check_compatibility=False,
            )
        return self._client

    def ensure_collection(self, embedding_dim: int | None = None) -> None:
        """Create collection if it does not exist. Recreate if dimensions mismatch."""
        dim = embedding_dim if embedding_dim is not None else self._vector_size
        client = self._get_client()
        collections = client.get_collections().collections
        names = [c.name for c in collections]
        if self._collection in names:
            try:
                info = client.get_collection(self._collection)
                vconfig = getattr(info.config.params, "vectors", None)
                current_size = None
                if hasattr(vconfig, "size"):
                    current_size = vconfig.size
                elif isinstance(vconfig, dict):
                    for v in vconfig.values():
                        if hasattr(v, "size"):
                            current_size = v.size
                            break
                if current_size is not None and current_size != dim:
                    logger.info("Collection %s has dim %d, need %d. Recreating.", self._collection, current_size, dim)
                    client.delete_collection(self._collection)
                elif current_size == dim:
                    logger.debug("Collection %s already exists", self._collection)
                    return
            except Exception as e:
                logger.warning("Could not check collection config: %s. Recreating.", e)
                client.delete_collection(self._collection)
        logger.info("Creating collection: %s (dim=%d)", self._collection, dim)
        client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """Upsert vectors with payloads."""
        client = self._get_client()
        points = [
            models.PointStruct(
                id=hash(ids[i]) % (2**63),
                vector=vectors[i],
                payload={
                    "content": payloads[i].get("content", ""),
                    "metadata": payloads[i].get("metadata", {}),
                },
            )
            for i in range(len(ids))
        ]
        client.upsert(collection_name=self._collection, points=points)
        logger.info("Upserted %d points to %s", len(ids), self._collection)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        client = self._get_client()
        results = client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        ).points
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "content": r.payload.get("content", ""),
                "metadata": r.payload.get("metadata", {}),
            }
            for r in results
        ]

    def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self._get_client().get_collections()
            return True
        except Exception as e:
            logger.warning("Qdrant health check failed: %s", e)
            return False


_vector_service: Optional[VectorService] = None


def get_vector_service() -> VectorService:
    """Get or create the vector service singleton."""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service
