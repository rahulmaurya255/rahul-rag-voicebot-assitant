"""Request and response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Text query request."""

    query: str = Field(..., min_length=1, max_length=2000, description="User question")


class QueryResponse(BaseModel):
    """Text query response."""

    answer: str = Field(..., description="RAG-generated answer")
    sources: list[str] = Field(default_factory=list, description="Source document references")


class VoiceQueryResponse(BaseModel):
    """Voice query response metadata (audio is streamed separately)."""

    text: str = Field(..., description="Transcribed user input")
    answer: str = Field(..., description="RAG-generated answer")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall status: healthy or degraded")
    api: bool = Field(..., description="API is up")
    qdrant: bool = Field(..., description="Qdrant is reachable")
    ollama: bool = Field(..., description="Ollama is reachable")


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code if applicable")
