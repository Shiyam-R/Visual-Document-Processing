"""
plain_ocr_baseline.py — Project 4 (Visual Document Processing)

The "no layout awareness at all" comparison point. Same OCR engine
(Tesseract, via labeled_lines.jsonl's underlying box.csv reference text)
and same per-line evaluation methodology as train_xgboost_baseline.py,
but classification uses ONLY text-pattern rules — no bounding box,
no position, no "which line is this" information whatsoever. This
isolates exactly what layout-awareness (hand-engineered in XGBoost,
learned in LayoutLM) actually buys you, holding OCR quality and
evaluation methodology constant.

Uses the SAME train/test receipt split as both other approaches
(receipt_ids from layoutlm_test.jsonl) for a fair comparison — this
script doesn't need "training" since the rules are hand-written, but
it's still evaluated ONLY on the held-out test receipts, exactly like
the other two.
"""

import json
import re
from pathlib import Path

from sklearn.metrics import classification_report

LABELED_LINES_PATH = Path("labeled_lines.jsonl")
LABEL_LIST = ["other", "company", "date", "total", "address"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}

DATE_PATTERN = re.compile(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}")
DECIMAL_PATTERN = re.compile(r"\d+\.\d{2}\b")
TOTAL_KEYWORDS = ("TOTAL", "AMOUNT", "JUMLAH", "BAYAR", "GRAND")
ADDRESS_KEYWORDS = ("JALAN", "TAMAN", "LOT", "NO.", "JLN", "PERSIARAN", "LORONG", "BAHRU", "SELANGOR")
COMPANY_SUFFIXES = ("SDN", "BHD", "ENTERPRISE", "RESTAURANT", "TRADING", "STORE", "MART")


def classify_line_text_only(text: str) -> str:
    """
    Pure text-pattern rules — no box, no position, no knowledge of
    where this line sits relative to any other line in the receipt.
    This is deliberately naive: it's the baseline layout-awareness is
    being compared AGAINST, not a competitive attempt in its own right.

    COMPANY_SUFFIXES checked BEFORE ADDRESS_KEYWORDS — found via a real
    bug: "BOOK TA .K (TAMAN DAYA) SDN BHD" (a company name referencing
    its own district) was matching ADDRESS_KEYWORDS' "TAMAN" first and
    returning "address" before ever reaching the SDN/BHD check, despite
    containing an unambiguous company-suffix. Company suffixes are legal
    entity markers unlikely to appear in a genuine street address, so
    checking them first is the more robust ordering — the same
    coincidental-place-name collision pattern found earlier in the
    fuzzy-matching labeling step, just resurfacing in a different rule.
    """
    text_upper = text.upper()
    if DATE_PATTERN.search(text):
        return "date"
    if DECIMAL_PATTERN.search(text) and any(k in text_upper for k in TOTAL_KEYWORDS):
        return "total"
    if any(k in text_upper for k in COMPANY_SUFFIXES):
        return "company"
    if any(k in text_upper for k in ADDRESS_KEYWORDS):
        return "address"
    return "other"


if __name__ == "__main__":
    with open("layoutlm_test.jsonl") as f:
        test_ids = set(json.loads(l)["receipt_id"] for l in f)
    print(f"Evaluating on {len(test_ids)} held-out test receipts (same split as XGBoost/LayoutLM)")

    y_true, y_pred = [], []
    with open(LABELED_LINES_PATH) as f:
        for raw in f:
            record = json.loads(raw)
            if record["receipt_id"] not in test_ids:
                continue
            y_true.append(LABEL_TO_ID[record["label"]])
            y_pred.append(LABEL_TO_ID[classify_line_text_only(record["text"])])

    print(f"Total test lines evaluated: {len(y_true)}\n")
    print("=== Per-class results (test set) — PLAIN OCR, no layout awareness ===")
    print(classification_report(y_true, y_pred, target_names=LABEL_LIST, zero_division=0))