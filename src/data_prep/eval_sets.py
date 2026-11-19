"""Đọc 4 bộ đánh giá của thiết kế 2 × 2 về một định dạng chung cho mọi notebook.

Mỗi câu: qid, dataset, document_id, image (đường dẫn tương đối từ gốc repo), question, gold, answers, answer_form,
evidence_bbox (chỉ bộ tiếng Việt tự xây), split.

- textvqa, docvqa, vitextvqa: data/qa/public/<bộ>.jsonl (src/data_prep/sample_public.py).
- visr_bench_vi: VISR_FILE (v2, src/data_prep/split_visr.py), kèm các trục độ khó để tách nhóm khi báo cáo.
  Câu không có split (bộ nhãn cũ) mang split "pilot": chỉ dùng chạy thử pipeline, không dùng chốt cấu hình.
"""

import json
import unicodedata
from pathlib import Path

from src.eval.score import answer_form

PUBLIC = ("textvqa", "docvqa", "vitextvqa")
VISR_FILE = Path("data/qa/visr_bench_vi_v2.jsonl")  # bộ nhãn cuối, đã chia dev/test (split_visr.py)
VISR_IMAGE_ROOT = Path("data/raw/visr_bench_vi")
ALL = PUBLIC + ("visr_bench_vi",)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def load(dataset: str, split: str) -> list[dict]:
    if dataset in PUBLIC:
        rows = [json.loads(l) for l in Path(f"data/qa/public/{dataset}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
        items = [{"qid": r["qid"], "dataset": dataset, "document_id": str(r["document_id"]), "image": r["image"],
                  "question": _nfc(r["question"]), "gold": _nfc(r["answer"]), "answers": [_nfc(a) for a in r["answers"]],
                  "answer_form": answer_form(r["answer"]), "evidence_bbox": None, "split": r["split"],
                  "dev_batch": r.get("dev_batch")} for r in rows]
    elif dataset == "visr_bench_vi":
        rows = [json.loads(l) for l in VISR_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        items = [{"qid": r["qid"], "dataset": dataset, "document_id": r["document_id"],
                  "image": (VISR_IMAGE_ROOT / r["image"]).as_posix(), "question": _nfc(r["question"]),
                  "gold": _nfc(r["answer"]), "answers": [_nfc(r["answer"])],
                  "answer_form": r.get("answer_form") or answer_form(r["answer"], bool(r.get("long_answer"))),
                  "evidence_bbox": r.get("evidence_bbox"), "question_type": r.get("question_type"),
                  "has_diacritic": r.get("has_diacritic"), "target_size": r.get("target_size"),
                  "eff_res": r.get("eff_res"), "evidence_type": r.get("evidence_type"),
                  "split": r.get("split") or "pilot"} for r in rows]
    else:
        raise ValueError(dataset)
    return [it for it in items if it["split"] == split]
