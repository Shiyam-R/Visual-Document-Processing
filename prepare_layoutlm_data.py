"""
prepare_layoutlm_data.py — Project 4 (Visual Document Processing)

Converts labeled_lines.jsonl (line-level labels from prepare_labels.py)
into word-level (word, normalized_bbox, label) records — what LayoutLM
actually consumes.

Two documented simplifications, not silently baked in:
1. box.csv gives LINE-level bounding boxes, not word-level. Each word in
   a line is assigned that LINE's box — an approximation (LayoutLM would
   ideally get each word's own tight box), acceptable given this is the
   granularity the source data provides.
2. Flat label scheme (O/COMPANY/DATE/TOTAL/ADDRESS), not BIO tagging.
   Justified because each receipt has at most one instance of each field
   (no repeated/interleaved entities to disambiguate) — BIO's B-/I-
   distinction exists to separate consecutive-but-distinct entity
   instances, which doesn't happen here.

Boxes normalized to LayoutLM's expected 0-1000 scale using each image's
OWN width/height (verified these vary significantly across receipts —
a fixed constant would silently corrupt every box).
"""

import json
import random
from pathlib import Path

from PIL import Image

DATA_DIR = Path("data")
LABELED_LINES_PATH = Path("labeled_lines.jsonl")
OUTPUT_TRAIN = Path("layoutlm_train.jsonl")
OUTPUT_TEST = Path("layoutlm_test.jsonl")

LABEL_LIST = ["other", "company", "date", "total", "address"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}

RANDOM_SEED = 42
TEST_FRACTION = 0.15


def normalize_box(x_min, y_min, x_max, y_max, width, height):
    return [
        max(0, min(1000, int(1000 * x_min / width))),
        max(0, min(1000, int(1000 * y_min / height))),
        max(0, min(1000, int(1000 * x_max / width))),
        max(0, min(1000, int(1000 * y_max / height))),
    ]


def build_receipt_record(receipt_id: str, lines: list[dict]) -> dict:
    img_path = DATA_DIR / "img" / f"{receipt_id}.jpg"
    with Image.open(img_path) as img:
        width, height = img.size

    words, boxes, labels = [], [], []
    for line in lines:
        box_norm = normalize_box(line["x_min"], line["y_min"], line["x_max"], line["y_max"], width, height)
        for word in line["text"].split():
            words.append(word)
            boxes.append(box_norm)
            labels.append(LABEL_TO_ID[line["label"]])

    return {"receipt_id": receipt_id, "words": words, "boxes": boxes, "labels": labels}


if __name__ == "__main__":
    lines_by_receipt: dict[str, list[dict]] = {}
    with open(LABELED_LINES_PATH) as f:
        for raw in f:
            record = json.loads(raw)
            lines_by_receipt.setdefault(record["receipt_id"], []).append(record)

    receipt_ids = sorted(lines_by_receipt.keys())
    print(f"Building word-level records for {len(receipt_ids)} receipts...")

    all_records = [build_receipt_record(rid, lines_by_receipt[rid]) for rid in receipt_ids]

    empty = [r["receipt_id"] for r in all_records if len(r["words"]) == 0]
    if empty:
        print(f"WARNING: {len(empty)} receipts produced zero words: {empty}")

    random.seed(RANDOM_SEED)
    shuffled = all_records[:]
    random.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * TEST_FRACTION))
    test_records = shuffled[:n_test]
    train_records = shuffled[n_test:]

    with open(OUTPUT_TRAIN, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(OUTPUT_TEST, "w") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    print(f"Train: {len(train_records)} receipts -> {OUTPUT_TRAIN}")
    print(f"Test:  {len(test_records)} receipts -> {OUTPUT_TEST}")
    print(f"Label scheme: {LABEL_TO_ID}")

    total_words = sum(len(r["words"]) for r in all_records)
    print(f"Total words across dataset: {total_words}")
    avg_words = total_words / len(all_records)
    print(f"Average words per receipt: {avg_words:.1f}")
