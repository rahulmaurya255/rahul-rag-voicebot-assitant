"""Document loaders for knowledge ingestion."""

from pathlib import Path
from typing import Iterator

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from app.utils.logging import get_logger

logger = get_logger(__name__)


def load_resume(path: str | Path) -> list[Document]:
    """Load resume from PDF or text file."""
    path = Path(path)
    if not path.exists():
        logger.warning("Resume path does not exist: %s", path)
        return []
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()
    if suffix in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()
    logger.warning("Unsupported resume format: %s", suffix)
    return []


def load_markdown(path: str | Path) -> list[Document]:
    """Load Markdown file(s) as text."""
    path = Path(path)
    if not path.exists():
        logger.warning("Markdown path does not exist: %s", path)
        return []
    if path.is_file():
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()
    if path.is_dir():
        loader = DirectoryLoader(str(path), glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        return loader.load()
    return []


def load_directory(path: str | Path) -> list[Document]:
    """Load all supported documents from a directory."""
    path = Path(path)
    if not path.is_dir():
        logger.warning("Not a directory: %s", path)
        return []
    docs: list[Document] = []
    for f in path.rglob("*"):
        if f.is_file():
            suffix = f.suffix.lower()
            if suffix == ".pdf":
                docs.extend(load_resume(f))
            elif suffix in (".md", ".markdown"):
                docs.extend(load_markdown(f))
            elif suffix == ".txt":
                loader = TextLoader(str(f), encoding="utf-8")
                docs.extend(loader.load())
    return docs


def load_all_sources(raw_dir: str | Path) -> Iterator[Document]:
    """Load all documents from data/raw directory."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        logger.warning("Raw data directory does not exist: %s", raw_dir)
        return
    for doc in load_directory(raw_dir):
        yield doc
