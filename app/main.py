"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.utils.logging import get_logger, setup_logging

settings = get_settings()
setup_logging("DEBUG" if settings.debug else "INFO")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # Warm-up: STT, TTS, LLM
    try:
        from app.voice.stt import warmup as stt_warmup
        from app.voice.tts import warmup as tts_warmup
        from app.services.llm_service import get_llm_service
        stt_warmup()
        tts_warmup()
        await get_llm_service().warmup()
    except Exception as e:
        logger.warning("Startup warm-up skipped or failed: %s", e)
    yield
    # Shutdown
    pass


app = FastAPI(
    title=settings.app_name,
    description="Production-grade RAG voice assistant for Rahul Maurya",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Rahul RAG Voice Assistant API", "docs": "/docs"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics placeholder."""
    return "# Metrics endpoint - add prometheus_client later"
