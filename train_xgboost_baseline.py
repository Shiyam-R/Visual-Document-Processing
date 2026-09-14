"""
train_xgboost_baseline.py — Project 4 (Visual Document Processing)

Hand-engineered spatial + text features per OCR LINE, fed to XGBoost —
the "layout-aware without a transformer" comparison point. Reuses
Project 1's exact toolkit. Uses the SAME train/test receipt split as
LayoutLM (from layoutlm_train.jsonl/layoutlm_test.jsonl) for a fair
comparison across all three approaches.

Feature set includes the total-vs-item-price fixes found during
development: decimal_rank_from_bottom_norm / is_last_decimal_line
(total is rarely literally the LAST decimal line — cash/change usually
follow it) and settlement_keyword_nearby (negative signal: CASH/CHANGE/
TENDER share total's surface features but aren't it).
"""

import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

DATA_DIR = Path("data")
LABELED_LINES_PATH = Path("labeled_lines.jsonl")
LABEL_LIST = ["other", "company", "date", "total", "address"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}

DATE_PATTERN = re.compile(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}")
DECIMAL_PATTERN = re.compile(r"\d+\.\d{2}\b")
CURRENCY_PATTERN = re.compile(r"RM|MYR|\$")
TOTAL_KEYWORDS = ("TOTAL", "AMOUNT", "JUMLAH", "BAYAR", "GRAND")
SETTLEMENT_KEYWORDS = ("CASH", "CHANGE", "TENDER", "BALANCE")
DATE_KEYWORDS = ("DATE", "TARIKH")
ADDRESS_KEYWORDS = ("JALAN", "TAMAN", "LOT", "NO.", "JLN", "PERSIARAN", "LORONG")


def load_labeled_lines() -> pd.DataFrame:
    records = []
    with open(LABELED_LINES_PATH) as f:
        for raw in f:
            records.append(json.loads(raw))
    return pd.DataFrame(records)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = []
    for receipt_id, group in df.groupby("receipt_id"):
        img_path = DATA_DIR / "img" / f"{receipt_id}.jpg"
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        group = group.reset_index(drop=True)
        n_lines = len(group)
        widths = (group["x_max"] - group["x_min"]).values
        max_width = widths.max() if widths.max() > 0 else 1

        has_decimal = [bool(DECIMAL_PATTERN.search(t)) for t in group["text"]]
        decimal_line_indices = [i for i, d in enumerate(has_decimal) if d]
        n_decimal_lines = len(decimal_line_indices)

        for i, row in group.iterrows():
            text = row["text"]
            text_upper = text.upper()
            x_min, x_max = row["x_min"], row["x_max"]
            y_min, y_max = row["y_min"], row["y_max"]
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

            prev_text = group["text"].iloc[i - 1].upper() if i > 0 else ""
            next_text = group["text"].iloc[i + 1].upper() if i < n_lines - 1 else ""
            total_keyword_adjacent = any(k in prev_text or k in next_text for k in TOTAL_KEYWORDS)
            settlement_keyword_nearby = any(
                k in text_upper or k in prev_text or k in next_text for k in SETTLEMENT_KEYWORDS
            )

            feature_rows.append({
                "receipt_id": receipt_id,
                "label": row["label"],
                "x_min_norm": x_min / img_width,
                "x_max_norm": x_max / img_width,
                "y_min_norm": y_min / img_height,
                "y_max_norm": y_max / img_height,
                "width_norm": width / img_width,
                "height_norm": height / img_height,
                "y_center_norm": (y_min + y_max) / 2 / img_height,
                "x_center_norm": (x_min + x_max) / 2 / img_width,
                "line_position_relative": i / max(n_lines - 1, 1),
                "width_ratio_to_widest": width / max_width,
                "text_length": len(text),
                "num_words": len(text.split()),
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

    return pd.DataFrame(feature_rows)


if __name__ == "__main__":
    print("Loading labeled lines...")
    df = load_labeled_lines()

    with open("layoutlm_train.jsonl") as f:
        train_ids = set(json.loads(l)["receipt_id"] for l in f)
    with open("layoutlm_test.jsonl") as f:
        test_ids = set(json.loads(l)["receipt_id"] for l in f)
    print(f"Reusing LayoutLM's split: {len(train_ids)} train receipts, {len(test_ids)} test receipts")

    print("Engineering features...")
    features_df = engineer_features(df)

    feature_cols = [c for c in features_df.columns if c not in ("receipt_id", "label")]
    features_df["label_id"] = features_df["label"].map(LABEL_TO_ID)

    train_df = features_df[features_df["receipt_id"].isin(train_ids)]
    test_df = features_df[features_df["receipt_id"].isin(test_ids)]
    print(f"Train lines: {len(train_df)}, Test lines: {len(test_df)}")

    X_train, y_train = train_df[feature_cols], train_df["label_id"]
    X_test, y_test = test_df[feature_cols], test_df["label_id"]

    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        objective="multi:softprob", num_class=len(LABEL_LIST),
        eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n=== Per-class results (test set) ===")
    print(classification_report(y_test, preds, target_names=LABEL_LIST, zero_division=0))

    print("=== Confusion matrix ===")
    cm = confusion_matrix(y_test, preds)
    print("Rows=actual, Cols=predicted:", LABEL_LIST)
    print(cm)

    print("\n=== Feature importance (top 10) ===")
    importance = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importance[:10]:
        print(f"  {feat}: {imp:.4f}")

    model.save_model("xgboost_baseline.json")
    print("\nSaved model to xgboost_baseline.json")