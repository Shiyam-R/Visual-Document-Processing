"""
app/services/extraction_service.py
─────────────────────────────────────────────────────────────────────────────
The actual extraction logic. Reuses predict_layoutlm.py's exact,
already-verified OCR + inference + confidence-aggregation approach —
this isn't new logic, just wired into the API + database instead of a
standalone script.
"""

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
from sqlalchemy.orm import Session

from app.config import MAX_LENGTH, MIN_CONFIDENCE
from app.db_models import Extraction
from app.exceptions import OCRError, PredictionError
from app.model_loader import LABEL_LIST, artifacts
from app.schemas.response import ExtractionResponse, FieldResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_box(x_min, y_min, x_max, y_max, width, height):
    return [
        max(0, min(1000, int(1000 * x_min / width))),
        max(0, min(1000, int(1000 * y_min / height))),
        max(0, min(1000, int(1000 * x_max / width))),
        max(0, min(1000, int(1000 * y_max / height))),
    ]


def run_ocr(image: Image.Image):
    width, height = image.size
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text or int(data["conf"][i]) < 0:
            continue
        x0, y0 = data["left"][i], data["top"][i]
        x1, y1 = x0 + data["width"][i], y0 + data["height"][i]
        words.append(text)
        boxes.append(normalize_box(x0, y0, x1, y1, width, height))
    return words, boxes


def extract_fields(image: Image.Image, filename: str, db: Session) -> ExtractionResponse:
    import torch

    words, boxes = run_ocr(image)
    if not words:
        raise OCRError("No text detected in the uploaded image.")

    try:
        encoding = artifacts.tokenizer(
            words, truncation=True, padding="max_length",
            max_length=MAX_LENGTH, is_split_into_words=True, return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        bbox_ids = [[0, 0, 0, 0] if w is None else boxes[w] for w in word_ids]
        bbox_tensor = torch.tensor([bbox_ids])

        with torch.no_grad():
            outputs = artifacts.model(**encoding, bbox=bbox_tensor)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_ids = torch.argmax(probs, dim=-1)
    except Exception as exc:
        logger.exception("Model inference failed")
        raise PredictionError(str(exc)) from exc

    # Aggregate word-level predictions into fields — same logic as
    # predict_layoutlm.py, already verified against real receipts.
    word_predictions = {}
    seen = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen:
            continue
        seen.add(word_idx)
        label_id = pred_ids[token_idx].item()
        confidence = probs[token_idx, label_id].item()
        word_predictions[word_idx] = (label_id, confidence)

    fields = {label: {"words": [], "confidences": []} for label in LABEL_LIST if label != "other"}
    for word_idx in sorted(word_predictions.keys()):
        label_id, confidence = word_predictions[word_idx]
        label = LABEL_LIST[label_id]
        if label == "other":
            continue
        fields[label]["words"].append(words[word_idx])
        fields[label]["confidences"].append(confidence)

    field_results = {}
    for label, data in fields.items():
        if not data["words"]:
            field_results[label] = {"value": None, "confidence": None, "flag": "not_found"}
        else:
            avg_conf = sum(data["confidences"]) / len(data["confidences"])
            field_results[label] = {
                "value": " ".join(data["words"]),
                "confidence": round(avg_conf, 3),
                "flag": "ok" if avg_conf >= MIN_CONFIDENCE else "low_confidence",
            }

    # Persist to the database
    record = Extraction(
        filename=filename,
        company_value=field_results["company"]["value"],
        company_confidence=field_results["company"]["confidence"],
        date_value=field_results["date"]["value"],
        date_confidence=field_results["date"]["confidence"],
        total_value=field_results["total"]["value"],
        total_confidence=field_results["total"]["confidence"],
        address_value=field_results["address"]["value"],
        address_confidence=field_results["address"]["confidence"],
        ocr_word_count=len(words),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info("Extraction saved: id=%s filename=%s", record.id, filename)

    return ExtractionResponse(
        id=record.id,
        filename=record.filename,
        uploaded_at=record.uploaded_at,
        company=FieldResult(**field_results["company"]),
        date=FieldResult(**field_results["date"]),
        total=FieldResult(**field_results["total"]),
        address=FieldResult(**field_results["address"]),
    )


def record_to_response(record: Extraction) -> ExtractionResponse:
    """Converts a stored DB row back into the same response shape —
    used by the GET endpoints, so /extract and /extractions/{id} return
    identically-shaped JSON."""
    return ExtractionResponse(
        id=record.id,
        filename=record.filename,
        uploaded_at=record.uploaded_at,
        company=FieldResult(
            value=record.company_value, confidence=record.company_confidence,
            flag="not_found" if record.company_value is None else (
                "ok" if (record.company_confidence or 0) >= MIN_CONFIDENCE else "low_confidence"
            ),
        ),
        date=FieldResult(
            value=record.date_value, confidence=record.date_confidence,
            flag="not_found" if record.date_value is None else (
                "ok" if (record.date_confidence or 0) >= MIN_CONFIDENCE else "low_confidence"
            ),
        ),
        total=FieldResult(
            value=record.total_value, confidence=record.total_confidence,
            flag="not_found" if record.total_value is None else (
                "ok" if (record.total_confidence or 0) >= MIN_CONFIDENCE else "low_confidence"
            ),
        ),
        address=FieldResult(
            value=record.address_value, confidence=record.address_confidence,
            flag="not_found" if record.address_value is None else (
                "ok" if (record.address_confidence or 0) >= MIN_CONFIDENCE else "low_confidence"
            ),
        ),
    )
