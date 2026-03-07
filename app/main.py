"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.utils.logging import get_logger, setup_logging

settings = get_settings()
setup_logging("DEBUG" if settings.debug else "INFO")
logger = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    try:
        from app.voice.stt import warmup as stt_warmup
        from app.voice.tts import warmup as tts_warmup
        from app.services.llm_service import get_llm_service
        await stt_warmup()
        await tts_warmup()
        await get_llm_service().warmup()
    except Exception as e:
        logger.warning("Startup warm-up skipped or failed: %s", e)
    yield


app = FastAPI(
    title=settings.app_name,
    description="RAG voice assistant",
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

# Serve frontend static files
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve React SPA - return index.html for all non-API routes."""
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    async def root():
        return {"message": "Mike Voice Assistant API", "docs": "/docs"}
