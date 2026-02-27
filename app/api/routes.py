"""API routes: /query, /voice-query, /health."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, PlainTextResponse

from app.core.config import get_settings
from app.models.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    VoiceQueryResponse,
)
from app.rag.chain import RAGChain
from app.services.vector_service import get_vector_service
from app.voice.stt import transcribe_bytes_async
from app.voice.tts import synthesize_async

router = APIRouter(prefix="/api", tags=["api"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Text query: RAG -> answer."""
    chain = RAGChain()
    answer, sources = await chain.query_full(request.query)
    return QueryResponse(answer=answer, sources=sources)


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Text query with streaming response."""
    chain = RAGChain()
    result = await chain.query(request.query, stream=True)
    if hasattr(result, "__aiter__"):
        async def gen():
            async for token in result:
                yield token
        return StreamingResponse(gen(), media_type="text/plain")
    return PlainTextResponse(result)


@router.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(audio: UploadFile = File(...)):
    """Voice query: audio -> STT -> RAG -> TTS -> return text + audio."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await audio.read()
    if len(content) > max_bytes:
        raise HTTPException(400, f"Audio too large. Max {settings.max_upload_size_mb}MB")
    if not content:
        raise HTTPException(400, "Empty audio file")

    text = await transcribe_bytes_async(content)
    if not text:
        return VoiceQueryResponse(
            text="",
            answer="I couldn't understand the audio. Please try again.",
        )

    chain = RAGChain()
    answer, _ = await chain.query_full(text)
    return VoiceQueryResponse(text=text, answer=answer)


@router.post("/voice-query/audio")
async def voice_query_audio(audio: UploadFile = File(...)):
    """Voice query returning audio response. Returns WAV stream."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await audio.read()
    if len(content) > max_bytes:
        raise HTTPException(400, f"Audio too large. Max {settings.max_upload_size_mb}MB")
    if not content:
        raise HTTPException(400, "Empty audio file")

    text = await transcribe_bytes_async(content)
    if not text:
        fallback = "I couldn't understand the audio. Please try again."
        wav_bytes = await synthesize_async(fallback)
        return StreamingResponse(
            iter([wav_bytes]),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=response.wav"},
        )

    chain = RAGChain()
    answer, _ = await chain.query_full(text)
    wav_bytes = await synthesize_async(answer)
    return StreamingResponse(
        iter([wav_bytes]),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=response.wav"},
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check: API, Qdrant, Ollama."""
    qdrant_ok = get_vector_service().health_check()
    ollama_ok = False
    try:
        import httpx
        settings = get_settings()
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass
    status = "healthy" if (qdrant_ok and ollama_ok) else "degraded"
    return HealthResponse(
        status=status,
        api=True,
        qdrant=qdrant_ok,
        ollama=ollama_ok,
    )
