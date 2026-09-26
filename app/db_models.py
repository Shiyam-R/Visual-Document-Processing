"""
app/db_models.py
─────────────────────────────────────────────────────────────────────────────
The one table this project needs: one row per uploaded document, storing
each extracted field alongside its confidence — not just the values, since
"confidence handling" is an explicit deliverable, not an afterthought.
"""

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.database import Base


class Extraction(Base):
    __tablename__ = "extractions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    company_value = Column(String, nullable=True)
    company_confidence = Column(Float, nullable=True)
    date_value = Column(String, nullable=True)
    date_confidence = Column(Float, nullable=True)
    total_value = Column(String, nullable=True)
    total_confidence = Column(Float, nullable=True)
    address_value = Column(String, nullable=True)
    address_confidence = Column(Float, nullable=True)

    ocr_word_count = Column(Integer, nullable=True)
