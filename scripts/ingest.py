"""Knowledge ingestion pipeline: load, chunk, embed, upsert to Qdrant."""

import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.loader import load_directory
from app.rag.splitter import chunk_documents
from app.rag.embeddings import get_embedding_service
from app.services.vector_service import get_vector_service


def main() -> None:
    """Run the full ingestion pipeline."""
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"

    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {raw_dir}. Add your documents (PDF, MD, TXT) and run again.")
        return

    # Load documents
    print("Loading documents from data/raw...")
    docs = load_directory(raw_dir)
    if not docs:
        print("No documents found in data/raw. Add PDF, MD, or TXT files.")
        return
    print(f"Loaded {len(docs)} documents.")

    # Chunk (200 chars, 30 overlap)
    print("Chunking (size=200, overlap=30)...")
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=30)
    if not chunks:
        print("No chunks produced.")
        return
    print(f"Produced {len(chunks)} chunks.")

    # Embed (OpenAI, batched with rate-limit handling)
    print("Embedding chunks via OpenAI (batches of 5)...")
    embedding_svc = get_embedding_service()
    texts = [c["content"] for c in chunks]
    vectors = embedding_svc.embed_texts(texts, is_query=False)
    print(f"Embedded {len(vectors)} chunks.")

    # Prepare payloads
    payloads = [
        {"content": c["content"], "metadata": c["metadata"]}
        for c in chunks
    ]
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Upsert to Qdrant
    print("Upserting to Qdrant Cloud...")
    vector_svc = get_vector_service()
    vector_svc.ensure_collection(embedding_dim=embedding_svc.dimension)
    vector_svc.upsert(ids=ids, vectors=vectors, payloads=payloads)

    print(f"Done. Ingested {len(chunks)} chunks into Qdrant.")


if __name__ == "__main__":
    main()
