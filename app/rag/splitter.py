"""Text chunking for RAG. Chunk size 200 chars, overlap 30."""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logging import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 200
CHUNK_OVERLAP = 30


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split documents into chunks with metadata.

    Returns chunks with: content, metadata (source, section, tags, category, importance).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = splitter.split_documents(documents)
    chunks: list[dict[str, Any]] = []
    for i, doc in enumerate(split_docs):
        meta = doc.metadata or {}
        chunk_meta = {
            "source": meta.get("source", "unknown"),
            "section": meta.get("section", ""),
            "tags": meta.get("tags", []),
            "category": meta.get("category", "general"),
            "importance": meta.get("importance", 1),
            "chunk_index": i,
        }
        chunks.append({"content": doc.page_content, "metadata": chunk_meta})
    logger.info("Chunked %d documents into %d chunks", len(documents), len(chunks))
    return chunks
