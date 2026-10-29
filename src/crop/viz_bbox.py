"""Vẽ hộp crop (từ dev_raw.jsonl) lên ảnh trang để kiểm tra rel_att có chỉ đúng chỗ không.

Chạy từ gốc repo:
    python -X utf8 -m src.crop.viz_bbox results/C3_vicrop/dev_raw.jsonl
Ảnh ra: results/C3_vicrop/viz/*.png (png đã gitignore).
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "visr_bench_vi"
FONT = "C:/Windows/Fonts/arial.ttf"
WIDTH = 800


def main() -> None:
    src = Path(sys.argv[1])
    out_dir = src.parent / "viz"
    out_dir.mkdir(exist_ok=True)
    labels = {}
    for l in (ROOT / "data/processed/visr_bench_vi/qa_vi_haiku_dev.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            d = json.loads(l)
            labels[d["qid"]] = d
    font = ImageFont.truetype(FONT, 22)

    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        img = Image.open(RAW / labels[r["qid"]]["image"]).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle(r["bbox"], outline=(255, 0, 0), width=8)
        s = WIDTH / img.width
        img = img.resize((WIDTH, int(img.height * s)))
        banner = Image.new("RGB", (WIDTH, 110), "white")
        bd = ImageDraw.Draw(banner)
        bd.text((8, 6), f"Q: {r['question'][:70]}", fill="black", font=font)
        bd.text((8, 38), f"gold: {r['gold'][:60]}", fill=(0, 120, 0), font=font)
        bd.text((8, 70), f"full: {r['pred_full'][:30]} | crop: {r['pred_crop'][:30]}", fill=(180, 0, 0), font=font)
        canvas = Image.new("RGB", (WIDTH, banner.height + img.height), "white")
        canvas.paste(banner, (0, 0))
        canvas.paste(img, (0, banner.height))
        canvas.save(out_dir / f"{r['qid'][-18:]}.png")
    print("Đã vẽ", len(list(out_dir.glob('*.png'))), "ảnh ->", out_dir)


if __name__ == "__main__":
    main()
