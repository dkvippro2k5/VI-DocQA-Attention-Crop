"""Đóng gói code + câu hỏi + ảnh của các bộ đánh giá thành một file zip để upload làm Kaggle dataset.

Chỉ đóng ảnh của split được chọn (mặc định dev và pilot). Ảnh test chỉ đóng gói khi chạy test một lần.
--diagnostic thêm văn bản và font của bộ ảnh chẩn đoán head (notebooks/havc_head_diagnosis.ipynb); ảnh chẩn đoán
được vẽ lại trên Kaggle nên không đóng gói.
Chạy từ gốc repo:
    python -m src.data_prep.pack_kaggle --name da2-eval --splits dev pilot
    python -m src.data_prep.pack_kaggle --name da2-havc --datasets --diagnostic
Đầu ra: data/processed/kaggle/<name>.zip (không commit).
"""

import argparse
import sys
import zipfile
from pathlib import Path

from src.data_prep import eval_sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="da2-eval")
    ap.add_argument("--splits", nargs="+", default=["dev", "pilot"])
    ap.add_argument("--datasets", nargs="*", default=list(eval_sets.ALL), help="để trống: không đóng câu hỏi, ảnh")
    ap.add_argument("--diagnostic", action="store_true", help="thêm data/diagnostic/corpus_*.jsonl và fonts/")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    out = Path("data/processed/kaggle") / f"{args.name}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = {p for p in Path("src").rglob("*.py") if "__pycache__" not in p.parts}
    files |= {Path(f"data/qa/public/{d}.jsonl") for d in eval_sets.PUBLIC if d in args.datasets}
    if "visr_bench_vi" in args.datasets:
        files.add(eval_sets.VISR_FILE)
    if args.diagnostic:
        files |= set(Path("data/diagnostic").glob("corpus_*.jsonl")) | set(Path("data/diagnostic/fonts").glob("*.ttf"))
    n_img = {}
    for d in args.datasets:
        imgs = {Path(it["image"]) for s in args.splits for it in eval_sets.load(d, s)}
        n_img[d] = len(imgs)
        files |= imgs
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # ảnh đã nén sẵn, không nén lại
        for f in sorted(files):
            z.write(f, f.as_posix())
    print(f"{out}: {len(files)} file, {out.stat().st_size / 1e6:.0f} MB; ảnh theo bộ: {n_img}")


if __name__ == "__main__":
    main()
