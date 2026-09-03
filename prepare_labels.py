"""
prepare_labels.py — Project 4 (Visual Document Processing)

SROIE only provides whole-field ground truth strings ("total": "9.00"),
not per-line/per-word labels. This matches each OCR line (from box/*.csv)
against the ground truth fields (key/*.json) via fuzzy matching, producing
a labeled dataset both the XGBoost baseline and the LayoutLM fine-tuning
(run separately in Codespaces) will share as a common input.

Real complications handled, not glossed over:
- Addresses span multiple consecutive OCR lines -> sliding-window matching.
- OCR misreads ground truth text (e.g. "D.I.Y." -> "D.T.Y.") -> fuzzy,
  not exact, matching, with punctuation/whitespace stripped before compare.
- date/total often appear as a substring within a longer line
  (e.g. "Rounded Total (RM): 9.00") -> partial-ratio matching.
"""

import json
import re
from pathlib import Path

from rapidfuzz import fuzz

DATA_DIR = Path("data")
OUTPUT_PATH = Path("labeled_lines.jsonl")

SINGLE_LINE_THRESHOLD = 75   # company, date, total
MULTI_LINE_THRESHOLD = 70    # address (harder — more room for OCR noise across lines)
MAX_ADDRESS_WINDOW = 4
MIN_CANDIDATE_LENGTH = 3     # guards against partial_ratio's failure mode on tiny
                             # lines (e.g. a lone "1" from a quantity column scores
                             # a spurious 100% "match" against any date/total that
                             # happens to contain that digit anywhere)


def normalize(s: str) -> str:
    """Strip everything except letters/digits, uppercase — makes spacing,
    punctuation, and minor formatting differences irrelevant to matching."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def parse_box_csv(path: Path) -> list[dict]:
    """box/*.csv format: 8 coords (4 corner points) + text, comma-separated.
    Text itself can legally contain commas, hence split(",", 8)."""
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split(",", 8)
        if len(parts) < 9:
            continue
        coords = list(map(int, parts[:8]))
        text = parts[8]
        xs = coords[0::2]
        ys = coords[1::2]
        lines.append({
            "text": text,
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
        })
    return lines


def match_single_line(target: str, ocr_lines: list[dict], threshold: int) -> tuple[int | None, int]:
    target_norm = normalize(target)
    if not target_norm:
        return None, 0
    best_score, best_idx = 0, None
    for i, line in enumerate(ocr_lines):
        line_norm = normalize(line["text"])
        if len(line_norm) < MIN_CANDIDATE_LENGTH:
            continue
        score = fuzz.partial_ratio(target_norm, line_norm)
        if score > best_score:
            best_score, best_idx = score, i
    return (best_idx, best_score) if best_score >= threshold else (None, best_score)


def match_multi_line(target: str, ocr_lines: list[dict], threshold: int, max_window: int) -> tuple[tuple[int, int] | None, int]:
    target_norm = normalize(target)
    if not target_norm:
        return None, 0
    best_score, best_range = 0, None
    for start in range(len(ocr_lines)):
        for window in range(1, max_window + 1):
            end = start + window
            if end > len(ocr_lines):
                break
            concat = "".join(normalize(ocr_lines[i]["text"]) for i in range(start, end))
            score = fuzz.ratio(target_norm, concat)
            if score > best_score:
                best_score, best_range = score, (start, end)
    return (best_range, best_score) if best_score >= threshold else (None, best_score)


def label_receipt(receipt_id: str) -> tuple[list[dict], dict]:
    ocr_lines = parse_box_csv(DATA_DIR / "box" / f"{receipt_id}.csv")
    key = json.loads((DATA_DIR / "key" / f"{receipt_id}.json").read_text())

    # idx -> (field, score) — track the BEST-scoring field per line, so a
    # loop-order collision (two fields matching the same line) resolves by
    # confidence, not by whichever field happened to run last.
    claims: dict[int, tuple[str, int]] = {}
    match_report = {}

    for field in ["company", "date", "total"]:
        if field not in key:
            match_report[field] = "missing_in_ground_truth"
            continue
        idx, score = match_single_line(key[field], ocr_lines, SINGLE_LINE_THRESHOLD)
        if idx is not None:
            if idx not in claims or score > claims[idx][1]:
                claims[idx] = (field, score)
            match_report[field] = f"matched (score={score})"
        else:
            match_report[field] = f"NO MATCH (best score={score})"

    labels = ["other"] * len(ocr_lines)
    for idx, (field, _score) in claims.items():
        labels[idx] = field

    if "address" in key:
        rng, score = match_multi_line(key["address"], ocr_lines, MULTI_LINE_THRESHOLD, MAX_ADDRESS_WINDOW)
        if rng is not None:
            for i in range(rng[0], rng[1]):
                if labels[i] == "other":  # don't overwrite an already-matched field
                    labels[i] = "address"
            match_report["address"] = f"matched lines {rng} (score={score})"
        else:
            match_report["address"] = f"NO MATCH (best score={score})"
    else:
        match_report["address"] = "missing_in_ground_truth"

    for line, label in zip(ocr_lines, labels, strict=False):
        line["label"] = label

    return ocr_lines, match_report


if __name__ == "__main__":
    receipt_ids = sorted(p.stem for p in (DATA_DIR / "key").glob("*.json"))
    print(f"Processing {len(receipt_ids)} receipts...")

    all_records = []
    field_match_counts = {"company": 0, "date": 0, "total": 0, "address": 0}
    fully_matched_receipts = 0
    failures = []

    for rid in receipt_ids:
        lines, report = label_receipt(rid)
        for line in lines:
            line["receipt_id"] = rid
        all_records.extend(lines)

        matched_fields = 0
        for field in ["company", "date", "total", "address"]:
            if report[field].startswith("matched"):
                field_match_counts[field] += 1
                matched_fields += 1
            elif "NO MATCH" in report[field]:
                failures.append((rid, field, report[field]))
        if matched_fields == 4:
            fully_matched_receipts += 1

    with open(OUTPUT_PATH, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(all_records)} labeled OCR lines to {OUTPUT_PATH}")
    print(f"\nField match rate (out of {len(receipt_ids)} receipts):")
    for field, count in field_match_counts.items():
        print(f"  {field}: {count} ({count/len(receipt_ids)*100:.1f}%)")
    print(f"\nReceipts with ALL 4 fields matched: {fully_matched_receipts} ({fully_matched_receipts/len(receipt_ids)*100:.1f}%)")
    print(f"\nTotal failures: {len(failures)}")
    print("First 10 failures (for spot-checking):")
    for rid, field, reason in failures[:10]:
        print(f"  {rid} / {field}: {reason}")
