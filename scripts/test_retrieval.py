"""Test vector retrieval from Qdrant with example queries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.embeddings import get_embedding_service
from app.services.vector_service import get_vector_service


def main() -> None:
    """Run example queries and print top chunks."""
    queries = [
        "What are Rahul's skills?",
        "Tell me about Namma Yatri experience",
        "What is Rahul's education?",
        "What ML projects has Rahul done?",
        "Rahul's experience at Bullsmart",
    ]

    embedding_svc = get_embedding_service()
    vector_svc = get_vector_service()

    print("Testing retrieval from Qdrant...\n")
    for q in queries:
        print(f"Query: {q}")
        print("-" * 60)
        vector = embedding_svc.embed_text(q, is_query=True)
        results = vector_svc.search(vector, top_k=3, score_threshold=0.5)
        for i, r in enumerate(results, 1):
            content = r.get("content", "")[:150] + "..." if len(r.get("content", "")) > 150 else r.get("content", "")
            score = r.get("score", 0)
            print(f"  [{i}] (score={score:.3f}) {content}")
        print()
    print("Retrieval test complete.")


if __name__ == "__main__":
    main()
