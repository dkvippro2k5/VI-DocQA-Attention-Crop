"""Tọa độ từng từ trên ảnh trang, dùng để gợi ý khung bằng chứng khi duyệt nhãn.

Nguồn: lớp text của PDF gốc (data/raw/visr_bench_vi/src_pdf/<doc[:8]>/doc.pdf), quy đổi từ điểm PDF sang pixel
ảnh trang. Tài liệu không có PDF thì dùng EasyOCR (khung theo cụm chữ, chia đều cho từng từ theo số ký tự,
nên chỉ gần đúng). Từ được chuẩn hóa bằng src.eval.score.normalize để khớp cùng cách với đáp án.
Kết quả OCR lưu đệm trong thư mục review để không phải chạy lại.
"""

import json
from pathlib import Path

from PIL import Image

from src.data_prep.label_visr_answers import IN_DIR
from src.eval.score import normalize

PDF_DIR = IN_DIR / "src_pdf"


def _tokens(text, box):
    """Một từ PDF có thể thành nhiều token sau chuẩn hóa ('DDR3_1' -> 'ddr3 1'); các token dùng chung khung."""
    return [[t, *box] for t in normalize(text).split()]


def pdf_words(pdf_path, page_id, img_size):
    import pymupdf
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_id]
        sx, sy = img_size[0] / page.rect.width, img_size[1] / page.rect.height
        out = []
        for x0, y0, x1, y1, text, *_ in page.get_text("words", sort=True):
            out += _tokens(text, [round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)])
        return out


_reader = None


def ocr_words(img_path):
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["vi"], gpu=False, verbose=False)
    import numpy as np
    with Image.open(img_path) as im:  # cv2 không mở được đường dẫn có ký tự tiếng Việt, nên đưa mảng ảnh
        arr = np.array(im.convert("RGB"))
    out = []
    for pts, text, _conf in _reader.readtext(arr):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        words = text.split()
        total = max(1, sum(len(w) for w in words) + len(words) - 1)
        pos = 0
        for w in words:
            a = x0 + (x1 - x0) * pos / total
            pos += len(w) + 1
            b = x0 + (x1 - x0) * min(1, (pos - 1) / total)
            out += _tokens(w, [round(a), round(y0), round(b), round(y1)])
    return out


GARBLED = set("˛˝˜ˆ˙°ˇ˘˚¸˝�")


def pdf_text_usable(words) -> bool:
    """Lớp text PDF chỉ dùng được khi có chữ và không bị lỗi font (font tiếng Việt cũ ra ký tự kiểu 'TR˛˝NG')."""
    if len(words) < 5:  # PDF scan: không có lớp text
        return False
    chars = "".join(w[0] for w in words)
    return sum(c in GARBLED for c in chars) / len(chars) < 0.005


def page_words(rows, cache_path: Path):
    """{đường dẫn ảnh tương đối trong IN_DIR: {"src": "pdf"|"ocr", "words": [[token, x0, y0, x1, y1], ...]}}."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    out = {}
    for doc_id, page_id, image in sorted({(r["document_id"], r["page_id"], r["image"]) for r in rows}):
        img_path = IN_DIR / image
        pdf = PDF_DIR / doc_id[:8] / "doc.pdf"
        words = None
        if pdf.exists():
            with Image.open(img_path) as im:
                words = pdf_words(pdf, page_id, im.size)
        if words is not None and pdf_text_usable(words):
            out[image] = {"src": "pdf", "words": words}
        else:
            if image not in cache:
                print(f"  OCR {image} ...")
                cache[image] = {"src": "ocr", "words": ocr_words(img_path)}
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            out[image] = cache[image]
    return out
