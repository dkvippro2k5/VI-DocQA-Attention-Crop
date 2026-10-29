"""Chấm lượt quét layer x kích thước hộp (dev_sweep.jsonl), in bảng theo cấu hình.

Chạy từ gốc repo:
    python -X utf8 -m src.eval.score_sweep results/C3_vicrop/dev_sweep.jsonl results/C3_vicrop/dev_raw.jsonl
Tham số 2 (tùy chọn): dev_raw.jsonl để in thêm dòng baseline ảnh đầy đủ.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from src.eval.score import contains_gold, exact_match, f1


def load(p: str):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    rows = load(sys.argv[1])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["layer"], r["bbox_size"])].append(r)

    def agg(items, col):
        n = len(items)
        return (n, sum(exact_match(i[col], i["gold"]) for i in items) / n,
                sum(contains_gold(i[col], i["gold"]) for i in items) / n,
                sum(f1(i[col], i["gold"]) for i in items) / n)

    print(f"{'layer':>5} {'bbox':>5} {'n':>3} {'EM':>6} {'contain':>8} {'F1':>6} {'area%':>6}")
    if len(sys.argv) > 2:
        n, em, ct, f = agg(load(sys.argv[2]), "pred_full")
        print(f"{'full':>5} {'-':>5} {n:>3} {em:6.3f} {ct:8.3f} {f:6.3f} {'100':>6}")
    for (layer, bs), items in sorted(groups.items()):
        n, em, ct, f = agg(items, "pred_crop")
        area = sum((i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]) /
                   (i["image_size"][0] * i["image_size"][1]) for i in items) / n * 100
        print(f"{layer:>5} {bs:>5} {n:>3} {em:6.3f} {ct:8.3f} {f:6.3f} {area:6.1f}")


if __name__ == "__main__":
    main()
