"""
app/schemas/response.py
─────────────────────────────────────────────────────────────────────────────
Response contracts. FieldResult mirrors predict_layoutlm.py's existing
{value, confidence, flag} shape — no reason to invent a new format for
a result we already validated works.
"""

from datetime import datetime

from pydantic import BaseModel


class FieldResult(BaseModel):
    value: str | None
    confidence: float | None
    flag: str  # "ok" | "low_confidence" | "not_found"


class ExtractionResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    company: FieldResult
    date: FieldResult
    total: FieldResult
    address: FieldResult

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
