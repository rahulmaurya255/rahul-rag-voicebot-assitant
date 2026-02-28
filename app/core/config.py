"""Application configuration via environment variables."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API
    app_name: str = Field(default="Rahul RAG Voice Assistant", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: str = Field(default="*", description="CORS allowed origins (comma-separated)")

    # Qdrant
    qdrant_url: str = Field(
        default="https://a436fd21-0d13-46e2-a95b-89bab4131236.us-west-2-0.aws.cloud.qdrant.io:6333",
        description="Qdrant server URL",
    )
    qdrant_collection: str = Field(default="rahul_knowledge", description="Vector collection name")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key (required for cloud)")

    # Embeddings (BGE default - no API key required)
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Sentence transformer model for embeddings",
    )
    embedding_dim: int = Field(default=384, description="Embedding vector dimension (BGE)")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    ollama_model: str = Field(
        default="mistral:7b-instruct",
        description="Ollama model name (e.g., mistral:7b-instruct, llama3.1:8b)",
    )
    ollama_timeout: float = Field(default=60.0, description="Ollama request timeout in seconds")

    # RAG
    rag_top_k: int = Field(default=5, description="Number of chunks to retrieve")
    rag_score_threshold: float = Field(default=0.3, description="Minimum similarity score for retrieval")

    # STT (Faster-Whisper)
    stt_model_size: str = Field(default="small", description="Whisper model size: tiny, base, small, medium, large")
    stt_device: str = Field(default="cpu", description="Device for STT: cpu or cuda")
    stt_compute_type: str = Field(default="int8", description="Compute type for CPU: int8 or float32")

    # TTS (Coqui)
    tts_model: str = Field(
        default="tts_models/en/ljspeech/tacotron2-DDC",
        description="Coqui TTS model name",
    )
    tts_device: str = Field(default="cpu", description="Device for TTS: cpu or cuda")

    # Security & Limits
    max_upload_size_mb: int = Field(default=10, description="Max audio upload size in MB")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute per IP")


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
