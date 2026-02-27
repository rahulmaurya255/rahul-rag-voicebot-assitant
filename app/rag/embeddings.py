"""Embedding service: OpenAI with BGE fallback when quota exceeded."""

import asyncio
import time
from typing import List, Union

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

OPENAI_BATCH_SIZE = 5
OPENAI_BATCH_DELAY_SEC = 1.0
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingService:
    """OpenAI embedding with BGE fallback on quota/rate errors."""

    def __init__(self) -> None:
        settings = get_settings()
        self._openai_client = None
        self._bge_model = None
        self._use_bge = False
        self._model = settings.embedding_model
        self._batch_size = OPENAI_BATCH_SIZE
        self._batch_delay = OPENAI_BATCH_DELAY_SEC
        if settings.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=settings.openai_api_key)
            except Exception:
                pass
        if not self._openai_client:
            self._use_bge = True
            logger.info("No OpenAI key, using BGE fallback")

    @property
    def dimension(self) -> int:
        """Embedding dimension (1536 for OpenAI, 384 for BGE)."""
        return 384 if self._use_bge else 1536

    def _get_bge(self):
        if self._bge_model is None:
            from sentence_transformers import SentenceTransformer
            self._bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._bge_model

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """Embed a single text."""
        if self._use_bge:
            model = self._get_bge()
            prefix = BGE_QUERY_PREFIX if is_query else ""
            return model.encode(prefix + text, normalize_embeddings=True).tolist()
        try:
            resp = self._openai_client.embeddings.create(model=self._model, input=text)
            return resp.data[0].embedding
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e).lower() or "rate" in str(e).lower():
                logger.warning("OpenAI quota/rate limit, switching to BGE: %s", e)
                self._use_bge = True
                return self.embed_text(text, is_query)
            raise

    def embed_texts(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Embed multiple texts."""
        if self._use_bge:
            model = self._get_bge()
            prefix = BGE_QUERY_PREFIX if is_query else ""
            prefixed = [prefix + t for t in texts]
            emb = model.encode(prefixed, batch_size=32, normalize_embeddings=True)
            return emb.tolist()
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            logger.info("Embedding batch %d/%d (%d texts)", i // self._batch_size + 1,
                        (len(texts) + self._batch_size - 1) // self._batch_size, len(batch))
            try:
                resp = self._openai_client.embeddings.create(model=self._model, input=batch)
                batch_emb = sorted(resp.data, key=lambda x: x.index)
                all_embeddings.extend([e.embedding for e in batch_emb])
            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e).lower() or "rate" in str(e).lower():
                    logger.warning("OpenAI quota/rate limit, switching to BGE: %s", e)
                    self._use_bge = True
                    return self.embed_texts(texts, is_query)
                raise
            if i + self._batch_size < len(texts):
                time.sleep(self._batch_delay)
        return all_embeddings

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
