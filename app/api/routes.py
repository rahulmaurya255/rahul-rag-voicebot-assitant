"""API routes: /query, /voice-query, /health."""

import json as _json

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse, PlainTextResponse

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
    """Voice query: audio -> STT -> RAG -> return text + answer."""
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


@router.post("/voice-query/stream")
async def voice_query_stream(audio: UploadFile = File(...)):
    """Voice query: audio -> STT -> RAG stream. Returns SSE."""
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    raw = await audio.read()
    if len(raw) > max_bytes:
        raise HTTPException(400, f"Audio too large. Max {settings.max_upload_size_mb}MB")
    if not raw:
        raise HTTPException(400, "Empty audio file")

    text = await transcribe_bytes_async(raw)
    if not text:
        fallback_msg = "I didn't quite catch that. Could you repeat your question?"
        transcription_evt = _json.dumps({"type": "transcription", "text": ""})
        token_evt = _json.dumps({"type": "token", "text": fallback_msg})
        done_evt = _json.dumps({"type": "done"})

        async def empty_gen():
            yield f"data: {transcription_evt}\n\n"
            yield f"data: {token_evt}\n\n"
            yield f"data: {done_evt}\n\n"

        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    chain = RAGChain()
    result = await chain.query(text, stream=True)

    transcription_evt = _json.dumps({"type": "transcription", "text": text})
    done_evt = _json.dumps({"type": "done"})

    async def gen():
        yield f"data: {transcription_evt}\n\n"
        if hasattr(result, "__aiter__"):
            async for token in result:
                evt = _json.dumps({"type": "token", "text": token})
                yield f"data: {evt}\n\n"
        else:
            evt = _json.dumps({"type": "token", "text": result})
            yield f"data: {evt}\n\n"
        yield f"data: {done_evt}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/voice-query/audio")
async def voice_query_audio(audio: UploadFile = File(...)):
    """Voice query returning MP3 audio response."""
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
        mp3_bytes = await synthesize_async(fallback)
        return StreamingResponse(
            iter([mp3_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=response.mp3"},
        )

    chain = RAGChain()
    answer, _ = await chain.query_full(text)
    mp3_bytes = await synthesize_async(answer)
    return StreamingResponse(
        iter([mp3_bytes]),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=response.mp3"},
    )


@router.post("/tts")
async def text_to_speech(request: QueryRequest):
    """Convert text to speech. Returns MP3 audio."""
    if not request.query.strip():
        raise HTTPException(400, "Empty text")
    mp3_bytes = await synthesize_async(request.query)
    return Response(content=mp3_bytes, media_type="audio/mpeg")


@router.post("/tts/sentence")
async def tts_sentence(request: QueryRequest):
    """TTS for a single sentence. Returns MP3 audio."""
    if not request.query.strip():
        raise HTTPException(400, "Empty text")
    mp3_bytes = await synthesize_async(request.query)
    return Response(content=mp3_bytes, media_type="audio/mpeg")


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check: API, Qdrant, Groq LLM."""
    qdrant_ok = get_vector_service().health_check()
    llm_ok = False
    try:
        from app.services.llm_service import get_llm_service
        llm_ok = await get_llm_service().warmup()
    except Exception:
        pass
    status = "healthy" if (qdrant_ok and llm_ok) else "degraded"
    return HealthResponse(
        status=status,
        api=True,
        qdrant=qdrant_ok,
        llm=llm_ok,
    )
