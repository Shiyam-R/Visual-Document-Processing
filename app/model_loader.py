"""
app/model_loader.py
─────────────────────────────────────────────────────────────────────────────
Loads the fine-tuned LayoutLM model + tokenizer exactly once at startup
(via main.py's lifespan), not per-request and not at import time — same
reasoning as the churn project's model_loader.py.
"""

from dataclasses import dataclass
from typing import Any

from app.config import MODEL_DIR
from app.exceptions import ArtifactLoadError
from app.utils.logger import get_logger

logger = get_logger(__name__)

LABEL_LIST = ["other", "company", "date", "total", "address"]


@dataclass
class ModelArtifacts:
    model: Any = None
    tokenizer: Any = None
    loaded: bool = False


artifacts = ModelArtifacts()


def load_artifacts() -> ModelArtifacts:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    logger.info("Loading LayoutLM model from %s ...", MODEL_DIR)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        model.eval()
    except Exception as exc:
        raise ArtifactLoadError(MODEL_DIR, str(exc)) from exc

    artifacts.model = model
    artifacts.tokenizer = tokenizer
    artifacts.loaded = True
    logger.info("Model loaded successfully.")
    return artifacts
