"""Tính trục độ khó, gán question_type / has_diacritic và chia dev/test cho VisR-Bench tiếng Việt.

Theo scope mục 7.1 và mục 9 (chốt 18/11/2026). Chỉ dùng nhãn đã duyệt, không dùng kết quả model.

Vào:
  data/qa/visr_bench_vi_r2.jsonl          bộ nhãn cuối (labeling_merge --round r2)
  --etype PATH (jsonl: qid, evidence_type) loại bằng chứng đã chốt sau khi duyệt đề xuất Codex; thay evidence_type
                                          của nháp Gemini cho các qid có trong file
Ra (khi không có --dry-run):
  data/qa/visr_bench_vi_v2.jsonl          mỗi câu thêm các trường trục, question_type, has_diacritic, split
  data/qa/visr_bench_vi_v2_split.json     cấu hình chia, danh sách tài liệu, phân bố dev/test, 10 cách gán tốt nhất

Cách chia: chia theo document_id. Với 19 tài liệu, duyệt hết 2^19 cách gán tài liệu vào dev, giữ các cách thỏa
ràng buộc (phần dev 27–33% số câu, dev ít nhất MIN_DEV_DOCS tài liệu, không tài liệu nào chiếm quá nửa test), rồi
chọn cách có tổng khoảng cách biến phân toàn phần (TVD) nhỏ nhất giữa dev và test trên ba phân bố: question_type,
answer_form, has_diacritic. Vì duyệt hết nên kết quả tất định, không cần seed; hòa điểm thì lấy cách có phần dev
gần 30% hơn, rồi tới mã nhị phân nhỏ hơn (tài liệu xếp theo document_id).

Chạy từ gốc repo:
    python -X utf8 -m src.data_prep.split_visr --dry-run                 # xem trước, không ghi file
    python -X utf8 -m src.data_prep.split_visr --etype PATH              # chốt: ghi visr_bench_vi_v2*
"""

import argparse
import json
import math
import sys
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "qa" / "visr_bench_vi_r2.jsonl"
OUT = ROOT / "data" / "qa" / "visr_bench_vi_v2.jsonl"
OUT_CFG = ROOT / "data" / "qa" / "visr_bench_vi_v2_split.json"

# Trục độ khó (scope mục 9). max_pixels giống mọi nhánh chạy Qwen2.5-VL.
MAX_PIXELS = 1024 * 28 * 28
SIZE_SMALL, SIZE_LARGE = 0.005, 0.05      # tỉ lệ diện tích vùng bằng chứng / trang
RES_LOW, RES_HIGH = 15, 25                # px, chiều cao khung thấp nhất sau khi thu ảnh

# Chia dev/test (scope mục 7.1).
DEV_TARGET, DEV_MIN, DEV_MAX = 0.30, 0.27, 0.33
MIN_DEV_DOCS = 5
BALANCE_KEYS = ("question_type", "answer_form", "has_diacritic")
LEVELS = {
    "question_type": ("small_detail", "table", "text"),
    "answer_form": ("short", "list", "long"),
    "has_diacritic": (True, False),
}


def has_diacritic(s: str) -> bool:
    """Có ít nhất một ký tự mang dấu tiếng Việt: dấu thanh, dấu mũ/móc/trăng (dấu tổ hợp sau NFD), hoặc đ."""
    s = unicodedata.normalize("NFC", s)
    if "đ" in s.lower():
        return True
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", s))


def target_box(boxes: list, mode: str) -> list:
    """Vùng đích: "all" lấy hộp bao các khung; "any" lấy khung nhỏ nhất."""
    if mode == "any":
        return min(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def axes(r: dict) -> dict:
    """Các trục độ khó của một câu, tính từ nhãn (bbox, kích thước ảnh, đáp án)."""
    W, H = r["image_size"]
    boxes, mode = r["evidence_bbox"], r.get("bbox_mode") or "all"
    t = target_box(boxes, mode)
    area = (t[2] - t[0]) * (t[3] - t[1]) / (W * H)
    size = "small" if area < SIZE_SMALL else ("large" if area > SIZE_LARGE else "medium")
    s = min(1.0, math.sqrt(MAX_PIXELS / (W * H)))
    cand = boxes if mode == "all" else [t]
    eff_px = min(b[3] - b[1] for b in cand) * s
    if r["answer_form"] == "short":
        res = "low" if eff_px < RES_LOW else ("high" if eff_px > RES_HIGH else "mid")
    else:
        res = None                         # chỉ áp dụng cho short (scope mục 9)
    return {"target_area_frac": round(area, 6), "target_size": size,
            "eff_text_px": round(eff_px, 2), "eff_res": res}


def question_type(r: dict, size: str) -> str:
    """Luật mục 9, theo thứ tự ưu tiên: table > small_detail (short, đích nhỏ) > text."""
    if r["evidence_type"] == "table":
        return "table"
    if r["answer_form"] == "short" and size == "small":
        return "small_detail"
    return "text"


def tvd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Khoảng cách biến phân toàn phần giữa các phân bố (hàng = cách gán, cột = mức)."""
    pa = a / np.maximum(a.sum(1, keepdims=True), 1)
    pb = b / np.maximum(b.sum(1, keepdims=True), 1)
    return 0.5 * np.abs(pa - pb).sum(1)


def dist(rows: list, key: str) -> dict:
    c = Counter(r[key] for r in rows)
    n = len(rows)
    return {str(v): {"n": c.get(v, 0), "pct": round(100 * c.get(v, 0) / n, 1) if n else 0.0} for v in LEVELS[key]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--etype", type=Path, default=None, help="jsonl qid + evidence_type đã chốt")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-missing-etype", action="store_true",
                    help="cho phép ghi khi còn câu chưa có evidence_type (không dùng cho bản chốt)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.etype:
        ov = {}
        for l in args.etype.read_text(encoding="utf-8").splitlines():
            if l.strip():
                x = json.loads(l)
                assert x["evidence_type"] in ("text", "table", "figure"), x
                ov[x["qid"]] = x["evidence_type"]
        unknown = set(ov) - {r["qid"] for r in rows}
        assert not unknown, f"qid không có trong bộ nhãn: {sorted(unknown)[:5]}"
        changed = 0
        for r in rows:
            if r["qid"] in ov:
                changed += r["evidence_type"] != ov[r["qid"]]
                r["evidence_type"] = ov[r["qid"]]
        print(f"Áp {len(ov)} loại bằng chứng từ {args.etype.name}, đổi {changed} câu")
    missing = [r["qid"] for r in rows if r["evidence_type"] is None]
    if missing:
        print(f"Cảnh báo: {len(missing)} câu chưa có evidence_type (coi là không phải table khi gán question_type)")
        if not args.dry_run and not args.allow_missing_etype:
            sys.exit("Dừng: cần --etype với loại bằng chứng đã chốt trước khi ghi bản chia.")

    for r in rows:
        r.update(axes(r))
        r["question_type"] = question_type(r, r["target_size"])
        r["has_diacritic"] = has_diacritic(r["answer"])

    # Ma trận đếm theo tài liệu: mỗi cột là một mức của một biến cần cân.
    docs = sorted({r["document_id"] for r in rows})
    di = {d: i for i, d in enumerate(docs)}
    cols = [(k, v) for k in BALANCE_KEYS for v in LEVELS[k]]
    M = np.zeros((len(docs), len(cols)), dtype=np.int64)
    n_doc = np.zeros(len(docs), dtype=np.int64)
    for r in rows:
        i = di[r["document_id"]]
        n_doc[i] += 1
        for k in BALANCE_KEYS:
            M[i, cols.index((k, r[k]))] += 1

    # Duyệt hết các cách gán: bit i = 1 nghĩa là tài liệu docs[i] vào dev.
    D = len(docs)
    masks = np.arange(1 << D, dtype=np.int64)
    S = ((masks[:, None] >> np.arange(D)) & 1).astype(np.int64)          # (2^D, D)
    n_dev = S @ n_doc
    frac = n_dev / n_doc.sum()
    n_dev_docs = S.sum(1)
    n_test = n_doc.sum() - n_dev
    max_doc_test = ((1 - S) * n_doc).max(1)
    ok = (frac >= DEV_MIN) & (frac <= DEV_MAX) & (n_dev_docs >= MIN_DEV_DOCS) & (max_doc_test <= n_test / 2)
    dev_c = S @ M
    test_c = M.sum(0) - dev_c
    score = np.zeros(len(masks))
    per_key = {}
    j = 0
    for k in BALANCE_KEYS:
        w = len(LEVELS[k])
        per_key[k] = tvd(dev_c[:, j:j + w], test_c[:, j:j + w])
        score += per_key[k]
        j += w
    idx = np.flatnonzero(ok)
    order = idx[np.lexsort((idx, np.abs(frac[idx] - DEV_TARGET), score[idx]))]
    best = int(order[0])

    def describe(m: int) -> dict:
        return {"mask": m, "dev_frac": round(float(frac[m]), 4), "n_dev": int(n_dev[m]),
                "n_dev_docs": int(n_dev_docs[m]), "score_tvd_sum": round(float(score[m]), 4),
                **{f"tvd_{k}": round(float(per_key[k][m]), 4) for k in BALANCE_KEYS},
                "dev_docs": [docs[i][:8] for i in range(D) if (m >> i) & 1]}

    dev_docs = {docs[i] for i in range(D) if (best >> i) & 1}
    for r in rows:
        r["split"] = "dev" if r["document_id"] in dev_docs else "test"
        r["version"] = "visr_bench_vi_v2"
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]

    def axis_table(sub: list) -> dict:
        sh = [r for r in sub if r["answer_form"] == "short"]
        return {
            "n": len(sub), "n_docs": len({r["document_id"] for r in sub}),
            "n_short": len(sh),
            **{k: dist(sub, k) for k in BALANCE_KEYS},
            "evidence_type": dict(Counter(str(r["evidence_type"]) for r in sub)),
            "target_size_short": dict(Counter(r["target_size"] for r in sh)),
            "eff_res_short": dict(Counter(r["eff_res"] for r in sh)),
            "short_small_and_diacritic": sum(r["target_size"] == "small" and r["has_diacritic"] for r in sh),
            "by_document": dict(Counter(r["document_id"][:8] for r in sub).most_common()),
        }

    cfg = {
        "version": "visr_bench_vi_v2", "created": date.today().isoformat(),
        "source": SRC.relative_to(ROOT).as_posix(),
        "etype_file": args.etype.as_posix() if args.etype else None,
        "n_missing_etype": len(missing),
        "axes": {"max_pixels": MAX_PIXELS, "size_bins": [SIZE_SMALL, SIZE_LARGE], "res_bins_px": [RES_LOW, RES_HIGH]},
        "split_rule": {"unit": "document_id", "search": f"exhaustive 2^{D}", "dev_target": DEV_TARGET,
                       "dev_range": [DEV_MIN, DEV_MAX], "min_dev_docs": MIN_DEV_DOCS,
                       "max_doc_share_of_test": 0.5, "objective": "sum TVD(dev, test) over " + ", ".join(BALANCE_KEYS),
                       "tie_break": "|dev_frac - 0.30|, then smaller mask", "n_feasible": int(ok.sum())},
        "chosen": describe(best),
        "top10": [describe(int(m)) for m in order[:10]],
        "dev_documents": sorted(dev_docs),
        "test_documents": sorted(set(docs) - dev_docs),
        "dev": axis_table(dev), "test": axis_table(test), "all": axis_table(rows),
    }

    print(f"Tài liệu {D}, cách gán khả thi {int(ok.sum())} / {1 << D}")
    print(f"Chọn: dev {len(dev)} câu ({100 * frac[best]:.1f}%), {len(cfg['dev']['by_document'])} tài liệu; "
          f"test {len(test)} câu, {cfg['test']['n_docs']} tài liệu; tổng TVD {score[best]:.4f}")
    for k in BALANCE_KEYS:
        print(f"  {k:14s} dev {cfg['dev'][k]}\n  {'':14s} test {cfg['test'][k]}")
    for part in ("dev", "test"):
        t = cfg[part]
        print(f"  {part}: short {t['n_short']}, đích nhỏ {t['target_size_short']}, độ phân giải {t['eff_res_short']}, "
              f"bằng chứng {t['evidence_type']}")
        print(f"        tài liệu {t['by_document']}")
    print("10 cách gán tốt nhất:")
    for c in cfg["top10"]:
        print(f"  TVD {c['score_tvd_sum']:.4f}  dev {100 * c['dev_frac']:.1f}%  {c['dev_docs']}")

    if args.dry_run:
        print("\n--dry-run: không ghi file.")
        return
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    OUT_CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nGhi: {OUT.relative_to(ROOT)} ({len(rows)}), {OUT_CFG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
