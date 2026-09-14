"""
prepare_layoutlm_data.py — Project 4 (Visual Document Processing)

CHANGED from the original version: each word in a line now gets a
PROPORTIONALLY SPLIT box (by character length) instead of the full
line's box duplicated across every word.

Why this matters, concretely: box.csv only gives LINE-level boxes, so
the original approach gave every word in a line the identical box. For
a company name that's a single wide line (very common — spans most of
the page width), every training example the model saw for "company" was
paired with one wide, IDENTICAL box shape. At real inference, Tesseract
gives each word its own much narrower individual box. The model likely
learned to associate "company" specifically with that wide shared-box
shape, not with narrow individual-word boxes — evidenced by: a genuine
TRAINING receipt still failing at 0.999 certainty on real inference
(ruling out a plain generalization gap), while multi-line fields like
address (which get naturally diverse box shapes across their several
lines even under the old approach) worked fine. This isn't a perfect
fix (character-length is an approximation — doesn't account for
variable glyph widths or spacing), but it's much closer to Tesseract's
real distribution than one identical box per line.
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


def split_line_box_by_words(words_in_line, x_min, x_max, y_min, y_max):
    """
    Proportionally splits a line's box across its words by character
    length. See module docstring for why this replaced giving every
    word in a line the identical full-line box.
    """
    if not words_in_line:
        return []
    total_chars = sum(len(w) for w in words_in_line) or 1
    line_width = x_max - x_min
    boxes = []
    cursor = x_min
    for w in words_in_line:
        word_width = line_width * (len(w) / total_chars)
        word_x_max = cursor + word_width
        boxes.append((cursor, y_min, word_x_max, y_max))
        cursor = word_x_max
    return boxes


def build_receipt_record(receipt_id: str, lines: list[dict]) -> dict:
    img_path = DATA_DIR / "img" / f"{receipt_id}.jpg"
    with Image.open(img_path) as img:
        width, height = img.size

    words, boxes, labels = [], [], []
    for line in lines:
        line_words = line["text"].split()
        word_boxes = split_line_box_by_words(
            line_words, line["x_min"], line["x_max"], line["y_min"], line["y_max"]
        )
        for word, (wx_min, wy_min, wx_max, wy_max) in zip(line_words, word_boxes, strict=True):
            box_norm = normalize_box(wx_min, wy_min, wx_max, wy_max, width, height)
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
    print(f"Average words per receipt: {total_words / len(all_records):.1f}")