"""Mô phỏng tài liệu scan để đo OCR khi KHÔNG có lớp text (pilot 38 câu, sổ tay GIGABYTE).

Ảnh PNG gốc được làm xuống cấp theo các mức (thu nhỏ độ phân giải, làm mờ, thêm nhiễu, nén JPEG),
rồi chỉ dùng EasyOCR. Bbox PyMuPDF trên PDF gốc làm mốc (đã kiểm tra bằng mắt trên trang 6, 9).

Đo theo từng mức:
- ocr_found: số câu OCR định vị được đáp án (khớp mờ, như pilot_bbox_check).
- iou_word_est: IoU giữa bbox OCR cấp từ ước lượng và bbox PyMuPDF.
- cer / cer_no_diacritics: tỉ lệ lỗi ký tự giữa đáp án và đoạn OCR khớp nhất, tính có dấu và sau
  khi bỏ dấu. Phần chênh lệch là lỗi do dấu thanh/dấu mũ (diacritic_gap).
- thời gian OCR mỗi trang.

Các mức xuống cấp là tham số tự chọn để mô phỏng, KHÔNG được hiệu chỉnh theo bản scan thật.
Chạy từ gốc repo:
    python -m src.data_prep.pilot_scan_sim
"""

import io
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

from src.data_prep.pilot_bbox_check import (
    DOC, ITEMS, OUT, PNG_WIDTH, RAW, iou, ocr_bbox, page_lines, pdf_bbox, strip_marks,
)

SEED = 0
# scale: tỉ lệ thu nhỏ so với PNG gốc (PNG ~200 dpi; 0.75 ~ 150 dpi, 0.5 ~ 100 dpi)
CONDITIONS = {
    "clean": None,
    "scan_light": {"scale": 0.75, "blur": 0.6, "noise": 4.0, "jpeg": 75},
    "scan_medium": {"scale": 0.6, "blur": 0.8, "noise": 6.0, "jpeg": 55},
    "scan_heavy": {"scale": 0.5, "blur": 1.0, "noise": 8.0, "jpeg": 40},
}


def degrade(img: Image.Image, cfg: dict, rng: np.random.Generator) -> Image.Image:
    w, h = img.size
    out = img.resize((int(w * cfg["scale"]), int(h * cfg["scale"])), Image.BILINEAR)
    out = out.filter(ImageFilter.GaussianBlur(cfg["blur"]))
    arr = np.asarray(out).astype(np.float32)
    arr = np.clip(arr + rng.normal(0, cfg["noise"], arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=cfg["jpeg"])
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def substring_edit_distance(pattern: str, text: str) -> int:
    """Khoảng cách sửa nhỏ nhất giữa pattern và một chuỗi con bất kỳ của text (căn chỉnh nửa toàn cục)."""
    if not pattern:
        return 0
    prev = [0] * (len(text) + 1)  # bắt đầu ở vị trí bất kỳ trong text không tốn chi phí
    for i, pc in enumerate(pattern, 1):
        cur = [i] + [0] * len(text)
        for j, tc in enumerate(text, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != tc))
        prev = cur
    return min(prev)


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


def line_text(segs, line_box, scale_back: float) -> str:
    """Ghép text các đoạn OCR cùng dòng với bbox dòng đã khớp (theo thứ tự trái sang phải)."""
    x0, y0, x1, y1 = line_box
    picked = []
    for poly, text, _ in segs:
        xs = [p[0] * scale_back for p in poly]
        ys = [p[1] * scale_back for p in poly]
        cy = (min(ys) + max(ys)) / 2
        if y0 - 5 <= cy <= y1 + 5 and max(xs) >= x0 - 5 and min(xs) <= x1 + 5:
            picked.append((min(xs), text))
    return " ".join(t for _, t in sorted(picked))


def scale_segs(segs, factor: float):
    return [[[[x * factor, y * factor] for x, y in poly], text, conf] for poly, text, conf in segs]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pdf = pymupdf.open(RAW / "src_pdf" / "0004e630" / "doc.pdf")
    pages = sorted({it[0] for it in ITEMS})
    lines = {p: page_lines(pdf[p]) for p in pages}

    import easyocr

    reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
    cache_dir = OUT / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results, per_item = {}, []
    for cond, cfg in CONDITIONS.items():
        rng = np.random.default_rng(SEED)
        scale = cfg["scale"] if cfg else 1.0
        ocr, times = {}, {}
        for p in pages:
            # Cache "clean" dùng chung tên file với pilot_bbox_check
            name = f"easyocr_{DOC}_p{p}.json" if cond == "clean" else f"easyocr_{DOC}_p{p}_{cond}.json"
            cache_file = cache_dir / name
            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                img = Image.open(RAW / "hf" / "Multilingual" / DOC / f"{DOC}_{p}.png").convert("RGB")
                if cfg:
                    img = degrade(img, cfg, rng)
                t0 = time.perf_counter()
                raw = reader.readtext(np.array(img))
                cached = {
                    "segs": [[[[float(x), float(y)] for x, y in poly], text, float(conf)] for poly, text, conf in raw],
                    "time_s": round(time.perf_counter() - t0, 2),
                }
                cache_file.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
            # Quy tọa độ OCR về hệ pixel của PNG gốc để so với mốc PyMuPDF
            ocr[p] = scale_segs(cached["segs"], 1 / scale)
            times[p] = cached["time_s"]

        found, ious, cers, cers_nd = 0, [], [], []
        for i, (p, q, a, qt, et, ctx) in enumerate(ITEMS, 1):
            page = pdf[p]
            _, ref = pdf_bbox(page, lines[p], a, ctx, PNG_WIDTH / page.rect.width)
            n, bb_line, bb_word = ocr_bbox(ocr[p], a, ctx)
            row = {"qid": f"{DOC}_p{p}_q{i}", "condition": cond, "answer": a, "question_type": qt,
                   "ocr_found": bb_line is not None, "iou_word_est": None, "cer": None, "cer_no_diacritics": None}
            if bb_line is not None:
                found += 1
                if ref:
                    row["iou_word_est"] = round(iou(ref, bb_word), 3)
                    ious.append(row["iou_word_est"])
                text = line_text(ocr[p], bb_line, 1.0)
                row["ocr_line_text"] = text
                ans = norm(a)
                row["cer"] = round(substring_edit_distance(ans, norm(text)) / len(ans), 3)
                row["cer_no_diacritics"] = round(
                    substring_edit_distance(strip_marks(ans), strip_marks(norm(text))) / len(ans), 3)
                cers.append(row["cer"])
                cers_nd.append(row["cer_no_diacritics"])
            per_item.append(row)

        mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
        results[cond] = {
            "params": cfg,
            "ocr_found": f"{found}/{len(ITEMS)}",
            "iou_word_est_mean": mean(ious),
            "iou_word_est_ge_0_5": sum(v >= 0.5 for v in ious),
            "cer_mean": mean(cers),
            "cer_no_diacritics_mean": mean(cers_nd),
            "diacritic_gap": round(mean(cers) - mean(cers_nd), 3) if cers else None,
            "exact_answer_read": sum(c == 0 for c in cers),
            "exact_answer_read_no_diacritics": sum(c == 0 for c in cers_nd),
            "time_ocr_per_page_s": times,
        }

    (OUT / "scan_sim_items.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per_item), encoding="utf-8")
    summary = {"seed": SEED, "n_items": len(ITEMS), "conditions": results}
    (OUT / "scan_sim_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
