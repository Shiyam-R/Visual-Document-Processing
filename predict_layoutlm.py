"""
predict_layoutlm.py — Project 4 (Visual Document Processing)

PARTIALLY TESTED: the Tesseract OCR portion (word extraction, boxes,
confidence) was verified in-sandbox against real SROIE receipt images.
The model-inference portion (load fine-tuned LayoutLM -> predict) could
NOT be tested here — it depends on layoutlm_finetuned/, which only
exists after running train_layoutlm.py in Codespaces. Verify this whole
pipeline end-to-end there before trusting its output.

Honest asymmetry worth knowing: training approximated each word's box as
its LINE's box (see prepare_layoutlm_data.py — box.csv only gives
line-level boxes). Tesseract's image_to_data gives genuine per-word
boxes at inference time — more precise than what the model trained on.
Shouldn't break anything (LayoutLM still learns a real word/position
relationship), but it's a real train/inference distribution difference,
not something to assume away.

Run:
    python predict_layoutlm.py path/to/receipt.jpg
"""

import sys

import pytesseract
import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = "layoutlm_finetuned"
LABEL_LIST = ["other", "company", "date", "total", "address"]
MAX_LENGTH = 512
MIN_CONFIDENCE = 0.5  # below this, a field is reported but flagged low-confidence


def normalize_box(x_min, y_min, x_max, y_max, width, height):
    return [
        max(0, min(1000, int(1000 * x_min / width))),
        max(0, min(1000, int(1000 * y_min / height))),
        max(0, min(1000, int(1000 * x_max / width))),
        max(0, min(1000, int(1000 * y_max / height))),
    ]


def run_ocr(image_path: str) -> tuple[list[str], list[list[int]], int, int]:
    img = Image.open(image_path)
    width, height = img.size
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words, boxes = [], []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text or int(data["conf"][i]) < 0:  # tesseract uses -1 for non-text regions
            continue
        x0, y0 = data["left"][i], data["top"][i]
        x1, y1 = x0 + data["width"][i], y0 + data["height"][i]
        words.append(text)
        boxes.append(normalize_box(x0, y0, x1, y1, width, height))

    return words, boxes, width, height


def predict(image_path: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    words, boxes, _, _ = run_ocr(image_path)
    if not words:
        return {"error": "No text detected by OCR", "fields": {}}

    encoding = tokenizer(
        words, boxes=boxes, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, is_split_into_words=True, return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)[0]  # (seq_len, num_labels)
        pred_ids = torch.argmax(probs, dim=-1)

    # Aggregate sub-token predictions back to WORDS, using only each
    # word's first sub-token (matches how labels were aligned in training
    # — continuation sub-tokens were never given a real label to learn).
    word_predictions = {}  # word_idx -> (label_id, confidence)
    seen_words = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen_words:
            continue
        seen_words.add(word_idx)
        label_id = pred_ids[token_idx].item()
        confidence = probs[token_idx, label_id].item()
        word_predictions[word_idx] = (label_id, confidence)

    # Group consecutive same-label words into field values.
    fields = {label: {"words": [], "confidences": []} for label in LABEL_LIST if label != "other"}
    for word_idx in sorted(word_predictions.keys()):
        label_id, confidence = word_predictions[word_idx]
        label = LABEL_LIST[label_id]
        if label == "other":
            continue
        fields[label]["words"].append(words[word_idx])
        fields[label]["confidences"].append(confidence)

    result = {}
    for label, data in fields.items():
        if not data["words"]:
            result[label] = {"value": None, "confidence": None, "flag": "not_found"}
        else:
            avg_conf = sum(data["confidences"]) / len(data["confidences"])
            result[label] = {
                "value": " ".join(data["words"]),
                "confidence": round(avg_conf, 3),
                "flag": "ok" if avg_conf >= MIN_CONFIDENCE else "low_confidence",
            }

    return {"fields": result}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_layoutlm.py path/to/receipt.jpg")
        sys.exit(1)

    import json
    result = predict(sys.argv[1])
    print(json.dumps(result, indent=2))
