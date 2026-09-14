"""
audit_company_predictions.py

Checks company prediction across genuine held-out test-set receipts.
Accepts an optional checkpoint path argument so you can spot-check an
IN-PROGRESS training run without waiting for all 5 epochs to finish —
e.g. after epoch 2 completes:

    python audit_company_predictions.py layoutlm_finetuned/checkpoint-266

If that argument is omitted, it loads the final model in
layoutlm_finetuned/ (only populated once training completes).
"""

import json
import sys

import pytesseract
import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "layoutlm_finetuned"
LABEL_LIST = ["other", "company", "date", "total", "address"]
MAX_LENGTH = 384
N_RECEIPTS_TO_CHECK = 10


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
    print(f"Loading model from: {MODEL_DIR}\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    with open("layoutlm_test.jsonl") as f:
        test_ids = [json.loads(line)["receipt_id"] for line in f]

    sample_ids = test_ids[:N_RECEIPTS_TO_CHECK]
    print(f"Checking {len(sample_ids)} GENUINE TEST-SET receipts (never trained on): {sample_ids}\n")

    results = []
    for rid in sample_ids:
        image_path = f"data/img/{rid}.jpg"
        try:
            with open(f"data/key/{rid}.json") as f:
                gt_company = json.load(f).get("company", "")
        except FileNotFoundError:
            gt_company = "?"

        words, boxes = run_ocr(image_path)
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
            pred_ids = torch.argmax(probs, dim=-1)

        company_words = []
        seen = set()
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen:
                continue
            seen.add(word_idx)
            if LABEL_LIST[pred_ids[token_idx].item()] == "company":
                company_words.append(words[word_idx])

        found = " ".join(company_words) if company_words else "(NOTHING PREDICTED AS COMPANY)"
        results.append((rid, gt_company, found))
        print(f"receipt={rid}  ground_truth={gt_company!r}")
        print(f"           predicted={found!r}\n")

    n_total_failures = sum(1 for _, _, found in results if "NOTHING" in found)
    print(f"\n=== SUMMARY: {n_total_failures}/{len(results)} test receipts had ZERO company words predicted ===")