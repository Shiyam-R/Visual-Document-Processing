"""
train_layoutlm.py — Project 4 (Visual Document Processing)

NOT TESTED IN THIS SESSION — HuggingFace Hub is blocked in this sandbox
(confirmed: 403, x-deny-reason: host_not_allowed), so this was written
carefully against the standard HF token-classification fine-tuning
recipe, but the actual pretrained-model-download + training loop has
never been executed. Run this in Codespaces and verify it end-to-end
before trusting the output.

Fine-tunes microsoft/layoutlm-base-uncased (text + 2D position, no
vision branch — the CPU/Codespaces-friendly choice over LayoutLMv3)
for token classification on layoutlm_train.jsonl / layoutlm_test.jsonl
(produced by prepare_layoutlm_data.py, already verified in-sandbox).

Run:
    pip install -r requirements-layoutlm.txt
    python train_layoutlm.py
"""

import json

import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/layoutlm-base-uncased"
LABEL_LIST = ["other", "company", "date", "total", "address"]
NUM_LABELS = len(LABEL_LIST)
OUTPUT_DIR = "layoutlm_finetuned"
MAX_LENGTH = 512


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def prepare_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_list(records)


def tokenize_and_align(examples, tokenizer):
    """
    Standard HF token-classification alignment: a word can split into
    multiple sub-word tokens. Only the FIRST sub-token of each word gets
    the real label; continuation sub-tokens and special tokens ([CLS],
    [SEP], padding) get -100, which PyTorch's loss functions are
    configured to ignore automatically.
    """
    tokenized = tokenizer(
        examples["words"],
        boxes=examples["boxes"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )

    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics(eval_pred):
    """
    Per-class precision/recall/F1, NOT raw accuracy — same reasoning as
    the churn project: ~89% of tokens are "other" (see prepare_labels.py
    output), so accuracy alone would be dominated by that majority class
    exactly the way churn's 73.5% non-churn majority made accuracy
    misleading there. The metric that matters is whether company/date/
    total/address are correctly identified, not overall token accuracy.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(predictions, labels, strict=False):
        for p, label in zip(pred_row, label_row, strict=False):
            if label != -100:
                true_labels.append(label)
                true_preds.append(p)

    report = classification_report(
        true_labels, true_preds, target_names=LABEL_LIST,
        output_dict=True, zero_division=0,
    )
    return {
        "company_f1": report["company"]["f1-score"],
        "date_f1": report["date"]["f1-score"],
        "total_f1": report["total"]["f1-score"],
        "address_f1": report["address"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
    }


if __name__ == "__main__":
    print(f"Loading tokenizer/model: {MODEL_NAME} (requires HF Hub access)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    train_records = load_jsonl("layoutlm_train.jsonl")
    test_records = load_jsonl("layoutlm_test.jsonl")
    print(f"Train: {len(train_records)} receipts | Test: {len(test_records)} receipts")

    train_ds = prepare_dataset(train_records).map(
        lambda ex: tokenize_and_align(ex, tokenizer), batched=True,
        remove_columns=["receipt_id", "words", "boxes"],
    )
    test_ds = prepare_dataset(test_records).map(
        lambda ex: tokenize_and_align(ex, tokenizer), batched=True,
        remove_columns=["receipt_id", "words", "boxes"],
    )

    # Small dataset (533 receipts) needs several epochs; batch size kept
    # small assuming CPU training in Codespaces — raise it if a GPU is
    # available (check with `python -c "import torch; print(torch.cuda.is_available())"`).
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=15,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("\nFinal evaluation:")
    print(trainer.evaluate())

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved fine-tuned model to {OUTPUT_DIR}/")
