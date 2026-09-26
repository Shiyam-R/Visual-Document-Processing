"""
app/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application factory.

Start the server:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Docs: http://localhost:8000/docs
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import API_DESCRIPTION, API_TITLE, API_VERSION
from app.database import Base, engine
from app.exceptions import DocumentAPIError
from app.model_loader import load_artifacts
from app.utils.logger import get_logger

logger = get_logger(__name__)

TAGS_METADATA = [
    {"name": "Info", "description": "Health and status."},
    {"name": "Extraction", "description": "Upload receipts, extract structured fields, retrieve past results."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("=" * 60)
    logger.info("Visual Document Processing API — starting up...")
    logger.info("=" * 60)

    logger.info("Creating database tables (if not already present)...")
    Base.metadata.create_all(bind=engine)

    try:
        load_artifacts()
        logger.info("Startup complete. API is ready to serve requests.")
    except Exception as exc:
        logger.critical("Startup failed: %s", exc, exc_info=True)
        raise

    yield

    logger.info("Visual Document Processing API — shutting down.")


def create_app() -> FastAPI:
    application = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=TAGS_METADATA,
    )

    @application.exception_handler(DocumentAPIError)
    async def document_error_handler(request: Request, exc: DocumentAPIError) -> JSONResponse:
        logger.warning("Handled error [%s] — %s", exc.status_code, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.message, "detail": exc.detail, "code": exc.status_code},
        )

    application.include_router(router)
    return application


app = create_app()
