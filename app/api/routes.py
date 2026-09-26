"""
app/api/routes.py
─────────────────────────────────────────────────────────────────────────────
Thin route handlers — actual logic lives in services/extraction_service.py.
"""

import io

from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import Extraction
from app.exceptions import ExtractionNotFoundError, InvalidImageError
from app.model_loader import artifacts
from app.schemas.response import ExtractionResponse, HealthResponse
from app.services.extraction_service import extract_fields, record_to_response

router = APIRouter()


@router.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract(file: UploadFile = File(...), db: Session = Depends(get_db)) -> ExtractionResponse:
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))  # re-open: verify() consumes the file pointer
    except UnidentifiedImageError as exc:
        raise InvalidImageError(f"Could not read '{file.filename}' as an image.") from exc

    return extract_fields(image, file.filename, db)


@router.get("/extractions", response_model=list[ExtractionResponse], tags=["Extraction"])
def list_extractions(db: Session = Depends(get_db), limit: int = 20) -> list[ExtractionResponse]:
    records = db.query(Extraction).order_by(Extraction.id.desc()).limit(limit).all()
    return [record_to_response(r) for r in records]


@router.get("/extractions/{extraction_id}", response_model=ExtractionResponse, tags=["Extraction"])
def get_extraction(extraction_id: int, db: Session = Depends(get_db)) -> ExtractionResponse:
    record = db.query(Extraction).filter(Extraction.id == extraction_id).first()
    if record is None:
        raise ExtractionNotFoundError(extraction_id)
    return record_to_response(record)


@router.get("/health", response_model=HealthResponse, tags=["Info"])
def health(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return HealthResponse(status="ok", model_loaded=artifacts.loaded, database_connected=db_ok)
