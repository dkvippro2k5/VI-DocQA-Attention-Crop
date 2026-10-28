"""Chia 23 labels thành dev/test theo document_id.

Dev: Doc B (13) + Doc C (8) = 21 items
Test: Doc A (2) = 2 items

Chạy từ gốc repo:
    python -m src.data_prep.split_dev_test
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "visr_bench_vi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    sys.stdout.reconfigure(encoding="utf-8")

    # Load all labels
    labels = []
    for f in ['qa_vi_haiku_pilot.jsonl', 'qa_vi_haiku_labels.jsonl']:
        path = OUT_DIR / f
        if path.exists():
            with open(path, encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        labels.append(json.loads(line))

    print(f"Loaded {len(labels)} total labels")

    # Test documents: 0004e63084e48e25a6982f11c7a77115 (2 items)
    test_doc_ids = ["0004e63084e48e25a6982f11c7a77115"]

    dev_labels = []
    test_labels = []

    for lbl in labels:
        if lbl["document_id"] in test_doc_ids:
            test_labels.append(lbl)
        else:
            dev_labels.append(lbl)

    # Add split field
    for lbl in dev_labels:
        lbl["split"] = "dev"
    for lbl in test_labels:
        lbl["split"] = "test"

    # Save dev
    dev_file = OUT_DIR / "qa_vi_haiku_dev.jsonl"
    with open(dev_file, "w", encoding="utf-8") as f:
        for lbl in dev_labels:
            f.write(json.dumps(lbl, ensure_ascii=False) + "\n")

    # Save test
    test_file = OUT_DIR / "qa_vi_haiku_test.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for lbl in test_labels:
            f.write(json.dumps(lbl, ensure_ascii=False) + "\n")

    print(f"\nDev: {len(dev_labels)} items")
    by_doc_dev = {}
    for lbl in dev_labels:
        doc_id = lbl["document_id"]
        by_doc_dev[doc_id] = by_doc_dev.get(doc_id, 0) + 1
    for doc_id in sorted(by_doc_dev.keys()):
        print(f"  {doc_id}: {by_doc_dev[doc_id]}")

    print(f"\nTest: {len(test_labels)} items")
    by_doc_test = {}
    for lbl in test_labels:
        doc_id = lbl["document_id"]
        by_doc_test[doc_id] = by_doc_test.get(doc_id, 0) + 1
    for doc_id in sorted(by_doc_test.keys()):
        print(f"  {doc_id}: {by_doc_test[doc_id]}")

    print(f"\n✓ Saved to {dev_file.name} and {test_file.name}")

if __name__ == "__main__":
    main()
