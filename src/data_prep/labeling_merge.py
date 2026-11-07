"""Gộp nhãn Gemini với quyết định duyệt tay thành bộ nhãn v1 (bằng chứng trong một trang).

Vào:
  data/processed/visr_bench_vi/qa_vi_visr_labels_external.jsonl   nhãn Gemini (labeling_validate.py)
  data/processed/visr_bench_vi/qa_vi_visr_answers.jsonl           đáp án Qwen3-VL (đối chiếu nguồn đáp án)
  data/processed/visr_bench_vi/review/review_decisions_v1.json    xuất từ review_v1.html
Ra:
  data/qa/visr_bench_vi_v1.jsonl          câu giữ lại (đúng/sửa), có evidence_bbox
  data/qa/visr_bench_vi_v1_dropped.jsonl  câu bỏ kèm lý do (để đếm giả định một trang sai ở đâu)
  data/qa/visr_bench_vi_v1_stats.json     thống kê duyệt cho báo cáo

answer_source: "gemini" (duyệt "đúng", hoặc "sửa" mà sau chuẩn hóa vẫn bằng đáp án Gemini),
"qwen" (đáp án cuối bằng đáp án Qwen sau chuẩn hóa), "human" (người duyệt tự viết).
bbox_mode: "all" = mỗi khung là một phần đáp án, "any" = các khung là chỗ thay thế. Quyết định xuất trước khi
trang có ô này thì coi là "all"; câu một ý mà có >= 2 khung được liệt kê để xem lại.
answer_form (short/long/list) tự gán theo src.eval.score.answer_form, không cần tick tay.
question_type và split để trống, gán ở bước sau. Bỏ qua tài liệu trong data/qa/excluded_documents.json.

Chạy từ gốc repo:
    python -X utf8 -m src.data_prep.labeling_merge              # vòng 1 -> visr_bench_vi_v1*
    python -X utf8 -m src.data_prep.labeling_merge --round r2   # vòng 2 -> visr_bench_vi_r2* (không có đáp án Qwen)
"""

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from src.data_prep.label_visr_answers import IN_DIR, excluded_docs, nfc
from src.eval.score import answer_form, normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "visr_bench_vi"
QWEN = DATA / "qa_vi_visr_answers.jsonl"
LABELS = {"v1": DATA / "qa_vi_visr_labels_external.jsonl", "r2": DATA / "qa_vi_visr_labels_external_r2.jsonl"}


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def answer_source(d, gemini, qwen):
    if d["verdict"] == "đúng":
        return "gemini"
    final = normalize(d["answer_final"])
    if gemini and final == normalize(gemini):
        return "gemini"
    if qwen and final == normalize(qwen):
        return "qwen"
    return "human"


def check_boxes(d, img_path):
    """Khung phải nằm trong ảnh và image_size phải khớp ảnh thật (khung lưu theo pixel ảnh gốc)."""
    with Image.open(img_path) as im:
        W, H = im.size
    problems = []
    if list(d.get("image_size") or []) != [W, H]:
        problems.append(f"image_size {d.get('image_size')} khác ảnh thật {[W, H]}")
    for b in d["evidence_bbox"]:
        x0, y0, x1, y1 = b
        if not (0 <= x0 < x1 <= W + 8 and 0 <= y0 < y1 <= H + 8):  # +8: đệm 4px của khung gợi ý
            problems.append(f"khung {b} ra ngoài ảnh {W}x{H}")
    return problems


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="v1", choices=sorted(LABELS))
    rnd = ap.parse_args().round
    DECISIONS = DATA / "review" / f"review_decisions_{rnd}.json"
    OUT = ROOT / "data" / "qa" / f"visr_bench_vi_{rnd}.jsonl"
    OUT_DROP = ROOT / "data" / "qa" / f"visr_bench_vi_{rnd}_dropped.jsonl"
    OUT_STATS = ROOT / "data" / "qa" / f"visr_bench_vi_{rnd}_stats.json"
    VERSION = f"visr_bench_vi_{rnd}"
    skip = excluded_docs()
    labels = {r["qid"]: r for r in load(LABELS[rnd]) if r["document_id"] not in skip}
    qwen = {r["qid"]: r["answer"] for r in load(QWEN)} if QWEN.exists() else {}
    exp = json.loads(DECISIONS.read_text(encoding="utf-8"))
    dec = {d["qid"]: d for d in exp["decisions"]}

    missing = sorted(set(labels) - set(dec))
    extra = sorted(set(dec) - set(labels))
    kept, dropped, to_check, problems = [], [], [], []
    for qid, r in labels.items():
        d = dec.get(qid)
        if d is None:
            continue
        if d["verdict"] == "bỏ câu":
            dropped.append({k: r[k] for k in ("qid", "document_id", "page_id", "image", "question")}
                           | {"gemini_answer": r["answer"], "drop_reason": d["drop_reason"], "note": d.get("note", "")})
            continue
        boxes = [[int(round(v)) for v in b] for b in d["evidence_bbox"]]
        for p in check_boxes(d, IN_DIR / r["image"]):
            problems.append(f"{qid}: {p}")
        answer = nfc(d["answer_final"]).strip()
        mode = d.get("bbox_mode") or "all"
        if len(boxes) < 2:
            mode = "all"
        if "bbox_mode" not in d and len(boxes) >= 2 and ";" not in answer:
            to_check.append(f"{qid} | {len(boxes)} khung | {answer[:60]}")
        kept.append({
            "qid": qid, "document_id": r["document_id"], "page_id": r["page_id"], "image": r["image"],
            "question": r["question"], "answer": answer,
            "evidence_bbox": boxes, "bbox_mode": mode, "image_size": d["image_size"],
            "evidence_type": r.get("evidence_type"), "question_type": None, "split": "",
            # dạng đáp án tự xác định (src.eval.score.answer_form); dấu "đáp án dài" người duyệt tick vẫn được tính
            "answer_form": answer_form(answer, bool(d.get("long_answer"))),
            "long_answer": bool(d.get("long_answer")),
            "answer_source": answer_source(d, r["answer"], qwen.get(qid)),
            "bbox_source": d.get("bbox_source") or "vẽ tay (trước khi có gợi ý)",
            "review_verdict": d["verdict"], "gemini_answer": r["answer"], "qwen_answer": qwen.get(qid),
            "review_note": d.get("note", ""), "reviewed_at": d.get("reviewed_at"), "version": VERSION,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept), encoding="utf-8")
    OUT_DROP.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in dropped), encoding="utf-8")

    by_doc = Counter(x["document_id"][:8] for x in kept)
    stats = {
        "version": VERSION, "scope": "bằng chứng trong một trang", "decisions_exported_at": exp.get("exported_at"),
        "n_labels": len(labels), "n_reviewed": len(labels) - len(missing), "n_kept": len(kept), "n_dropped": len(dropped),
        "verdict": dict(Counter(dec[q]["verdict"] for q in labels if q in dec)),
        "answer_source": dict(Counter(x["answer_source"] for x in kept)),
        "bbox_source": dict(Counter(x["bbox_source"] for x in kept)),
        "bbox_mode": dict(Counter(x["bbox_mode"] for x in kept)),
        "n_boxes": dict(sorted(Counter(len(x["evidence_bbox"]) for x in kept).items())),
        "long_answer": sum(x["long_answer"] for x in kept),
        "answer_form": dict(Counter(x["answer_form"] for x in kept)),
        "drop_reason": dict(Counter(x["drop_reason"] for x in dropped)),
        "by_document": dict(by_doc),
        "gemini_accept_rate": round(sum(x["answer_source"] == "gemini" for x in kept) / max(1, len(labels) - len(missing)), 3),
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if missing:
        print(f"\nChưa duyệt: {len(missing)} câu, ví dụ {missing[:3]}")
    if extra:
        print(f"Quyết định không có trong nhãn (tài liệu bị loại?): {len(extra)}")
    if problems:
        print("\nLỗi khung:")
        for p in problems:
            print("  " + p)
    if to_check:
        print(f"\nDuyệt trước khi có ô 'chỗ thay thế', một ý mà >= 2 khung (đang coi là 'all'): {len(to_check)}")
        for t in to_check:
            print("  " + t)
    print(f"\nGhi: {OUT.relative_to(ROOT)} ({len(kept)}), {OUT_DROP.relative_to(ROOT)} ({len(dropped)}), "
          f"{OUT_STATS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
