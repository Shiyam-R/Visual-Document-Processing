"""
debug_predict.py — diagnostic for the company=null issue

Prints EVERY word with its full probability distribution across all 5
labels, not just the aggregated final result. This will show definitively
whether:
(a) the model confidently predicts "other" for company words (genuine
    model/training problem), or
(b) it predicts "company" but something in aggregation is dropping it
    (a bug in predict_layoutlm.py we haven't found yet), or
(c) word/box alignment is shifted (predictions look like they belong to
    the WRONG word entirely — e.g. "BOOK" shows what looks like a
    date-line prediction)

Run: python debug_predict.py data/img/000.jpg
"""

import sys

import pytesseract
import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = "layoutlm_finetuned"
LABEL_LIST = ["other", "company", "date", "total", "address"]
MAX_LENGTH = 384


def normalize_box(x_min, y_min, x_max, y_max, width, height):
    return [
        max(0, min(1000, int(1000 * x_min / width))),
        max(0, min(1000, int(1000 * y_min / height))),
        max(0, min(1000, int(1000 * x_max / width))),
        max(0, min(1000, int(1000 * y_max / height))),
    ]


def run_ocr(image_path):
    img = Image.open(image_path)
    width, height = img.size
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
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


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/img/000.jpg"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    words, boxes = run_ocr(image_path)
    print(f"Total OCR words detected: {len(words)}")
    print(f"MAX_LENGTH: {MAX_LENGTH} -- {'TRUNCATION RISK' if len(words) > MAX_LENGTH else 'no truncation risk'}")
    print()

    encoding = tokenizer(
        words, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, is_split_into_words=True, return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)

    bbox_ids = [[0, 0, 0, 0] if w is None else boxes[w] for w in word_ids]
    bbox_tensor = torch.tensor([bbox_ids])

    with torch.no_grad():
        outputs = model(**encoding, bbox=bbox_tensor)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    print(f"{'word':<15} {'box':<20} {'pred':<10} " + " ".join(f"{l:>10}" for l in LABEL_LIST))
    print("-" * 100)

    seen = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen:
            continue
        seen.add(word_idx)
        word = words[word_idx]
        box = boxes[word_idx]
        p = probs[token_idx]
        pred_label = LABEL_LIST[torch.argmax(p).item()]
        prob_str = " ".join(f"{p[i].item():>10.3f}" for i in range(len(LABEL_LIST)))
        # Only print the first 40 words to keep output readable —
        # company/date typically appear in this range
        if word_idx < 40:
            print(f"{word:<15} {str(box):<20} {pred_label:<10} {prob_str}")