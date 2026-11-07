"""Chấm lượt quét cấu hình x kích thước hộp (dev_sweep.jsonl), in bảng theo cấu hình.

Cấu hình là `layer` (C3) hoặc `mode` (C4/C5: chỉ ảnh crop / ảnh gốc + crop).

Chạy từ gốc repo:
    python -X utf8 -m src.eval.score_sweep results/C3_vicrop/dev_sweep.jsonl results/C3_vicrop/dev_raw.jsonl
    python -X utf8 -m src.eval.score_sweep results/C5_havc_vi/dev_sweep.jsonl
Tham số 2 (tùy chọn): file có cột pred_full để in thêm dòng baseline ảnh đầy đủ; bỏ trống thì lấy pred_full
trong chính file quét nếu có.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from src.eval.score import answer_form, contains_gold, exact_match, f1, item_cover


def load(p: str):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    rows = load(sys.argv[1])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["layer"] if "layer" in r else r["mode"], r["bbox_size"])].append(r)

    def agg(items, col):
        n = len(items)
        return (n, sum(exact_match(i[col], i["gold"]) for i in items) / n,
                sum(contains_gold(i[col], i["gold"]) for i in items) / n,
                sum(f1(i[col], i["gold"]) for i in items) / n,
                sum(item_cover(i[col], i["gold"]) for i in items) / n)

    w = max(5, *(len(str(c)) for c, _ in groups))
    forms = {}
    for r in rows:
        forms[answer_form(r["gold"])] = forms.get(answer_form(r["gold"]), set()) | {r["qid"]}
    print("Dạng đáp án (số câu):", {k: len(v) for k, v in sorted(forms.items())})
    print(f"{'cfg':>{w}} {'bbox':>5} {'n':>3} {'EM':>6} {'contain':>8} {'F1':>6} {'item':>6} {'area%':>6}")
    full = load(sys.argv[2]) if len(sys.argv) > 2 else list({r["qid"]: r for r in rows if "pred_full" in r}.values())
    if full:
        n, em, ct, f, it = agg(full, "pred_full")
        print(f"{'full':>{w}} {'-':>5} {n:>3} {em:6.3f} {ct:8.3f} {f:6.3f} {it:6.3f} {'100':>6}")
    for (cfg, bs), items in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        n, em, ct, f, it = agg(items, "pred_crop")
        area = sum((i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]) /
                   (i["image_size"][0] * i["image_size"][1]) for i in items) / n * 100
        print(f"{cfg:>{w}} {bs:>5} {n:>3} {em:6.3f} {ct:8.3f} {f:6.3f} {it:6.3f} {area:6.1f}")


if __name__ == "__main__":
    main()
