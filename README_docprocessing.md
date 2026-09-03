# Visual Document Processing — Receipt Field Extraction

Project 4 (Phase 2, DL Foundations): extracts structured fields
(company, date, total, address) from scanned receipts. Two approaches
built for direct comparison — plain OCR vs. a fine-tuned layout-aware
transformer.

Dataset: [SROIE 2019](https://github.com/zzzDavid/ICDAR-2019-SROIE) —
626 real scanned receipts with ground-truth field labels.

## Setup

```bash
# Dataset (not committed here — fetched fresh from source)
git clone https://github.com/zzzDavid/ICDAR-2019-SROIE.git
mv ICDAR-2019-SROIE/data ./data
rm -rf ICDAR-2019-SROIE

# Dependencies
sudo apt-get install -y tesseract-ocr
pip install -r requirements-layoutlm.txt
```

## Pipeline

```bash
# 1. Label each OCR line with its field type (fuzzy-matched against
#    ground truth — verified 624/626 receipts fully matched; two known,
#    accepted edge cases: receipt 033 has a blank ground-truth total,
#    receipt 485 has a genuine date/total tie on a 3-character line)
python prepare_labels.py

# 2. Convert to word-level, box-normalized records for LayoutLM
python prepare_layoutlm_data.py

# 3. Fine-tune microsoft/layoutlm-base-uncased (NOT verified locally —
#    HuggingFace Hub is blocked in the environment this was built in;
#    verify this step actually works before trusting its output)
python train_layoutlm.py

# 4. Run inference on a real receipt
python predict_layoutlm.py data/img/000.jpg
```

## Model choice

LayoutLM v1 (`microsoft/layoutlm-base-uncased`), not v3 — text + 2D
position embeddings only, no vision-transformer branch. Lighter and
more CPU/Codespaces-friendly, while teaching the same core mechanism:
attention over spatial position, not just token sequence.
