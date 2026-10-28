"""Gán nhãn 137 QA pairs từ VisR-Bench VI - phiên bản đơn giản.

Sử dụng questions từ multilingual_QA.json, answers từ OCR trên ảnh.
Không gọi API (tránh timeout).

Chạy từ gốc repo:
    python -m src.data_prep.label_visr_vi_simple
"""

import json
import sys
import os
from pathlib import Path
import unicodedata
import numpy as np

from PIL import Image
import easyocr

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data" / "raw" / "visr_bench_vi"
OUT_DIR = ROOT / "data" / "processed" / "visr_bench_vi"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def norm_text(s: str) -> str:
    """Chuẩn hóa text: NFC, lowercase, trim whitespace."""
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    # Load multilingual QA
    with open(IN_DIR / "multilingual_QA.json", encoding="utf-8") as f:
        all_docs = json.load(f)

    # Only use 5 documents that have images downloaded
    downloaded_doc_ids = [
        "0004e63084e48e25a6982f11c7a77115",
        "01d7b45bc59e9652f1764e6fbec2f05e",
        "1aa33928936bdf5e0989361a3a68a3a0",
        "9d418db8080ff2ea5053886d4cfdffaf",
        "eb55d9dbe2e5ccaf00e6c15fe8191a98",
    ]

    print(f"Loaded {len(all_docs)} documents from multilingual_QA.json")

    # Extract VI QA pairs from downloaded documents only
    vi_items = []
    for doc in all_docs:
        doc_id = doc["file_name"]
        if doc_id not in downloaded_doc_ids:
            continue
        for qa_idx, qa in enumerate(doc["qa_list"]):
            if qa.get("detected_language") == "vi" and qa.get("question"):
                vi_items.append({
                    "document_id": doc_id,
                    "page_id": qa.get("page_index", 0),
                    "question": qa["question"],
                })

    print(f"Extracted {len(vi_items)} VI questions from {len(downloaded_doc_ids)} documents")

    # Initialize OCR reader
    reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)

    # Generate labels
    labeled = []
    for idx, item in enumerate(vi_items):
        qid = f"visr_vi_{idx:05d}"
        doc_id = item["document_id"]
        page_id = item["page_id"]
        question = item["question"]

        print(f"[{idx+1}/{len(vi_items)}] {qid}: {question[:60]}...", end=" ")

        # Find image path
        img_path = IN_DIR / "hf" / "Multilingual" / doc_id / f"{doc_id}_{page_id}.png"
        if not img_path.exists():
            print("❌ image not found")
            continue

        # Load and OCR
        try:
            img = Image.open(img_path).convert("RGB")
            ocr_result = reader.readtext(np.array(img))
            ocr_text = " ".join([text for _, text, _ in ocr_result])
        except Exception as e:
            print(f"❌ OCR failed: {e}")
            continue

        # Create label record
        label = {
            "qid": qid,
            "document_id": doc_id,
            "page_id": page_id,
            "image": str(Path("hf") / "Multilingual" / doc_id / f"{doc_id}_{page_id}.png"),
            "question": norm_text(question),
            "answer": norm_text(ocr_text)[:100],  # First 100 chars of OCR
            "evidence_bbox": None,
            "question_type": "small_detail",  # Default
            "split": "dev",
            "annotator_model": "ocr_easyocr",
        }

        labeled.append(label)
        print(f"✓ ({len(ocr_text)} chars)")

        # Checkpoint every 50 items
        if (idx + 1) % 50 == 0:
            print(f"\n  [Checkpoint {idx+1} items]\n")

    # Save all items
    out_file = OUT_DIR / "qa_vi_haiku_full.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for lbl in labeled:
            f.write(json.dumps(lbl, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n✓ Labeled {len(labeled)} / {len(vi_items)} items")
    print(f"✓ Saved to {out_file.relative_to(ROOT)}")

    # Stats
    by_doc = {}
    for lbl in labeled:
        doc_id = lbl["document_id"]
        by_doc[doc_id] = by_doc.get(doc_id, 0) + 1

    print("\nBy document:")
    for doc_id in sorted(downloaded_doc_ids):
        count = by_doc.get(doc_id, 0)
        print(f"  {doc_id}: {count}")


if __name__ == "__main__":
    main()
