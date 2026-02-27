"""Initialize Qdrant collection. Run before first ingest."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.vector_service import get_vector_service


def main() -> None:
    """Create the knowledge collection if it does not exist."""
    vs = get_vector_service()
    vs.ensure_collection()
    print("Collection ready.")


if __name__ == "__main__":
    main()
