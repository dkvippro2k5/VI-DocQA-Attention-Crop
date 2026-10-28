"""Đo lỗi dấu tiếng Việt trên toàn bộ nhãn (pilot + new).

Sử dụng OCR trên ảnh sạch, so sánh với PyMuPDF. Tính:
- CER (Character Error Rate) với dấu
- CER bỏ dấu
- Diacritic gap = (CER với dấu) - (CER bỏ dấu) -> % lỗi chỉ do dấu

Chạy từ gốc repo:
    python -m src.data_prep.measure_diacritic_error
"""

import json
import sys
import unicodedata
from pathlib import Path
from collections import Counter

import pymupdf
import easyocr
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data" / "raw" / "visr_bench_vi"
OUT_DIR = ROOT / "data" / "processed" / "visr_bench_vi"


def strip_marks(s: str) -> str:
    """Bỏ dấu thanh/mũ (NFD -> bỏ combining marks -> NFC)."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


def substring_edit_distance(pattern: str, text: str) -> int:
    """Semi-global edit distance."""
    if not pattern:
        return 0
    prev = [0] * (len(text) + 1)
    for i, pc in enumerate(pattern, 1):
        cur = [i] + [0] * len(text)
        for j, tc in enumerate(text, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != tc))
        prev = cur
    return min(prev)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    # Load all labels
    labels = []
    for jsonl_path in sorted(OUT_DIR.glob("qa_vi_haiku*.jsonl")):
        # Skip bbox-only files (pilot20, bbox_pilot)
        if "bbox" in jsonl_path.name:
            continue
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                labels.append(json.loads(line))

    print(f"Loaded {len(labels)} labels from {OUT_DIR.name}")

    # Group by document to cache PDF
    by_doc = {}
    for label in labels:
        doc_id = label["document_id"]
        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(label)

    # Initialize OCR reader
    reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)

    results = []
    doc_pdfs = {}

    for doc_id, doc_labels in by_doc.items():
        # Load PDF
        pdf_path = IN_DIR / "src_pdf" / doc_id / "doc.pdf"
        if pdf_path.exists():
            doc_pdfs[doc_id] = pymupdf.open(pdf_path)

        for label in doc_labels:
            page_id = label["page_id"]
            answer = label["answer"]

            if not answer:
                continue

            # Load PNG
            img_path = IN_DIR / label["image"]
            if not img_path.exists():
                print(f"  {label['qid']}: image not found, skip")
                continue

            # OCR on clean PNG
            try:
                img = Image.open(img_path).convert("RGB")
                ocr_result = reader.readtext(np.array(img))
            except Exception as e:
                print(f"  {label['qid']}: OCR failed: {e}")
                continue

            # Extract text from OCR
            ocr_text = " ".join([text for _, text, _ in ocr_result])

            # Compute metrics
            ans_norm = norm(answer)
            ocr_norm = norm(ocr_text)

            cer = substring_edit_distance(ans_norm, ocr_norm) / len(ans_norm) if ans_norm else 0
            cer_nd = substring_edit_distance(strip_marks(ans_norm), strip_marks(ocr_norm)) / len(ans_norm) if ans_norm else 0
            diacritic_gap = cer - cer_nd

            results.append({
                "qid": label["qid"],
                "document_id": doc_id,
                "page_id": page_id,
                "answer": answer,
                "question_type": label.get("question_type"),
                "cer": round(cer, 3),
                "cer_no_diacritics": round(cer_nd, 3),
                "diacritic_gap": round(diacritic_gap, 3),
                "ocr_text_sample": ocr_norm[:100],
            })

    # Summary statistics
    cer_vals = [r["cer"] for r in results]
    cer_nd_vals = [r["cer_no_diacritics"] for r in results]
    gap_vals = [r["diacritic_gap"] for r in results]

    # Split by question_type
    by_type = {}
    for r in results:
        qtype = r.get("question_type", "unknown")
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(r)

    summary = {
        "n_items": len(results),
        "cer_mean": round(sum(cer_vals) / len(cer_vals) if cer_vals else 0, 3),
        "cer_no_diacritics_mean": round(sum(cer_nd_vals) / len(cer_nd_vals) if cer_nd_vals else 0, 3),
        "diacritic_gap_mean": round(sum(gap_vals) / len(gap_vals) if gap_vals else 0, 3),
        "diacritic_gap_pct": round(100 * sum(gap_vals) / sum(cer_vals) if cer_vals and sum(cer_vals) > 0 else 0, 1),
        "by_question_type": {},
    }

    for qtype in sorted(by_type.keys()):
        rows = by_type[qtype]
        cer_t = [r["cer"] for r in rows]
        gap_t = [r["diacritic_gap"] for r in rows]
        summary["by_question_type"][qtype] = {
            "n": len(rows),
            "cer_mean": round(sum(cer_t) / len(cer_t) if cer_t else 0, 3),
            "diacritic_gap_mean": round(sum(gap_t) / len(gap_t) if gap_t else 0, 3),
        }

    out_summary = OUT_DIR / "diacritic_error_summary.json"
    out_items = OUT_DIR / "diacritic_error_items.jsonl"

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_items, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSummary saved to {out_summary.relative_to(ROOT)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
