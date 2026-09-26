"""
app/config.py
─────────────────────────────────────────────────────────────────────────────
Central config. DATABASE_URL comes from Render's env var in production;
defaults to a local SQLite file for convenient local testing without
needing Render connectivity for basic development.
"""

import os

API_TITLE = "Visual Document Processing API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Extracts structured fields (company, date, total, address) from "
    "uploaded receipt images using a fine-tuned LayoutLM model, with "
    "per-field confidence, and persists results to Postgres."
)

MODEL_DIR = os.getenv("MODEL_DIR", "layoutlm_finetuned")
MAX_LENGTH = 384
MIN_CONFIDENCE = 0.5  # below this, a field is flagged low_confidence, not dropped

# Render's Postgres connection strings sometimes use the legacy
# "postgres://" scheme (Heroku-style) — SQLAlchemy 1.4+ requires
# "postgresql://". Converting defensively rather than assuming Render's
# exact format, since this is a well-known, common gotcha.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_dev.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
