"""
predict_baseline.py — Project 4 (Visual Document Processing)

Shared prediction script for BOTH plain-OCR and XGBoost approaches —
they operate at the same granularity (per OCR LINE, reconstructed via
Tesseract's block/par/line grouping), unlike LayoutLM which works at
the WORD level and needs predict_layoutlm.py's separate neural-network
path. This does OCR once, reconstructs lines, engineers the same
features train_xgboost_baseline.py used, and can classify with either
method — producing the SAME structured JSON shape as predict_layoutlm.py
so all three approaches are directly comparable on the same real image.

Run:
    python predict_baseline.py data/img/000.jpg plain
    python predict_baseline.py data/img/000.jpg xgboost
"""

import json
import re
import sys

import pytesseract
from PIL import Image
from xgboost import XGBClassifier

LABEL_LIST = ["other", "company", "date", "total", "address"]

DATE_PATTERN = re.compile(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}")
DECIMAL_PATTERN = re.compile(r"\d+\.\d{2}\b")
CURRENCY_PATTERN = re.compile(r"RM|MYR|\$")
TOTAL_KEYWORDS = ("TOTAL", "AMOUNT", "JUMLAH", "BAYAR", "GRAND")
SETTLEMENT_KEYWORDS = ("CASH", "CHANGE", "TENDER", "BALANCE")
DATE_KEYWORDS = ("DATE", "TARIKH")
ADDRESS_KEYWORDS = ("JALAN", "TAMAN", "LOT", "NO.", "JLN", "PERSIARAN", "LORONG", "BAHRU", "SELANGOR")
COMPANY_SUFFIXES = ("SDN", "BHD", "ENTERPRISE", "RESTAURANT", "TRADING", "STORE", "MART")


def reconstruct_lines(image_path: str) -> tuple[list[dict], int, int]:
    """Groups Tesseract's word-level output into LINES via its own
    block/par/line hierarchy — same granularity box.csv originally gave."""
    img = Image.open(image_path)
    width, height = img.size
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    lines_by_key = {}
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text or int(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x0, y0 = data["left"][i], data["top"][i]
        x1, y1 = x0 + data["width"][i], y0 + data["height"][i]
        if key not in lines_by_key:
            lines_by_key[key] = {"words": [], "x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1}
        entry = lines_by_key[key]
        entry["words"].append(text)
        entry["x_min"] = min(entry["x_min"], x0)
        entry["x_max"] = max(entry["x_max"], x1)
        entry["y_min"] = min(entry["y_min"], y0)
        entry["y_max"] = max(entry["y_max"], y1)

    lines = [
        {"text": " ".join(v["words"]), "x_min": v["x_min"], "x_max": v["x_max"],
         "y_min": v["y_min"], "y_max": v["y_max"]}
        for v in lines_by_key.values()
    ]
    return lines, width, height


def classify_line_text_only(text: str) -> str:
    """Same rules as plain_ocr_baseline.py — zero spatial information.
    COMPANY_SUFFIXES checked before ADDRESS_KEYWORDS (see that file's
    docstring for the real bug this fixes — a company name referencing
    its own district was misclassified as address)."""
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


def engineer_features_for_lines(lines: list[dict], img_width: int, img_height: int) -> list[dict]:
    """Same feature set as train_xgboost_baseline.py's engineer_features,
    applied to a single receipt's freshly-OCR'd lines instead of the
    pre-labeled training data."""
    n_lines = len(lines)
    widths = [line["x_max"] - line["x_min"] for line in lines]
    max_width = max(widths) if widths and max(widths) > 0 else 1

    has_decimal = [bool(DECIMAL_PATTERN.search(line["text"])) for line in lines]
    decimal_line_indices = [i for i, d in enumerate(has_decimal) if d]
    n_decimal_lines = len(decimal_line_indices)

    feature_rows = []
    for i, line in enumerate(lines):
        text = line["text"]
        text_upper = text.upper()
        x_min, x_max = line["x_min"], line["x_max"]
        y_min, y_max = line["y_min"], line["y_max"]
        width = x_max - x_min
        height = y_max - y_min
        alpha_chars = [c for c in text if c.isalpha()]

        if has_decimal[i] and n_decimal_lines > 0:
            rank_from_bottom = decimal_line_indices[::-1].index(i)
            decimal_rank_norm = rank_from_bottom / max(n_decimal_lines - 1, 1)
            is_last_decimal = rank_from_bottom == 0
        else:
            decimal_rank_norm = -1
            is_last_decimal = False

        prev_text = lines[i - 1]["text"].upper() if i > 0 else ""
        next_text = lines[i + 1]["text"].upper() if i < n_lines - 1 else ""
        total_keyword_adjacent = any(k in prev_text or k in next_text for k in TOTAL_KEYWORDS)
        settlement_keyword_nearby = any(
            k in text_upper or k in prev_text or k in next_text for k in SETTLEMENT_KEYWORDS
        )

        feature_rows.append({
            "x_min_norm": x_min / img_width, "x_max_norm": x_max / img_width,
            "y_min_norm": y_min / img_height, "y_max_norm": y_max / img_height,
            "width_norm": width / img_width, "height_norm": height / img_height,
            "y_center_norm": (y_min + y_max) / 2 / img_height,
            "x_center_norm": (x_min + x_max) / 2 / img_width,
            "line_position_relative": i / max(n_lines - 1, 1),
            "width_ratio_to_widest": width / max_width,
            "text_length": len(text), "num_words": len(text.split()),
            "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
            "has_date_pattern": bool(DATE_PATTERN.search(text)),
            "has_decimal_number": has_decimal[i],
            "has_currency_symbol": bool(CURRENCY_PATTERN.search(text_upper)),
            "has_total_keyword": any(k in text_upper for k in TOTAL_KEYWORDS),
            "has_date_keyword": any(k in text_upper for k in DATE_KEYWORDS),
            "has_address_keyword": any(k in text_upper for k in ADDRESS_KEYWORDS),
            "uppercase_ratio": (sum(c.isupper() for c in alpha_chars) / len(alpha_chars)) if alpha_chars else 0,
            "starts_with_digit": text[0].isdigit() if text else False,
            "decimal_rank_from_bottom_norm": decimal_rank_norm,
            "is_last_decimal_line": is_last_decimal,
            "total_keyword_adjacent": total_keyword_adjacent,
            "settlement_keyword_nearby": settlement_keyword_nearby,
        })
    return feature_rows


def build_result(lines: list[dict], predicted_labels: list[str]) -> dict:
    fields = {label: [] for label in LABEL_LIST if label != "other"}
    for line, label in zip(lines, predicted_labels, strict=True):
        if label != "other":
            fields[label].append(line["text"])
    return {label: (" ".join(texts) if texts else None) for label, texts in fields.items()}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("plain", "xgboost"):
        print("Usage: python predict_baseline.py <image_path> <plain|xgboost>")
        sys.exit(1)

    image_path, method = sys.argv[1], sys.argv[2]
    lines, img_width, img_height = reconstruct_lines(image_path)

    if method == "plain":
        predicted = [classify_line_text_only(line["text"]) for line in lines]
    else:
        model = XGBClassifier()
        model.load_model("xgboost_baseline.json")
        feature_rows = engineer_features_for_lines(lines, img_width, img_height)
        import pandas as pd
        X = pd.DataFrame(feature_rows)
        pred_ids = model.predict(X)
        predicted = [LABEL_LIST[p] for p in pred_ids]

    result = build_result(lines, predicted)
    print(json.dumps({"method": method, "fields": result}, indent=2))