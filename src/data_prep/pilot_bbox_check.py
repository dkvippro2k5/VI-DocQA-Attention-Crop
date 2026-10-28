"""Đo chất lượng bbox của bộ pilot (sổ tay GIGABYTE GA-P55-UD6, các trang index 3, 4, 6, 9).

Hai nguồn bbox:
- PyMuPDF: tìm đáp án trên PDF gốc bằng page.search_for (chỉ dùng được khi có lớp text sạch).
- EasyOCR (vi+en): tìm đoạn OCR chứa đáp án sau khi bỏ dấu; ảnh là nguồn đúng, OCR chỉ định vị.

Khi đáp án xuất hiện nhiều lần trên trang, mỗi câu có thể khai báo `ctx` (một cụm chữ nằm trong
cùng dòng với đáp án) để chọn đúng chỗ ở cả hai nguồn.

Đo:
- số lần khớp (hits) từng nguồn; câu khớp >1 mà không có ctx bị coi là mơ hồ.
- iou_word_est: IoU giữa bbox PyMuPDF và bbox OCR cấp từ ƯỚC LƯỢNG (EasyOCR chỉ trả bbox cả đoạn,
  nên chia chiều ngang theo vị trí ký tự; xấp xỉ, không phải bbox từ thật).
- thời gian: PyMuPDF (s/câu), OCR (s/trang, không tính tải model).

Không có bộ vàng độc lập, nên đây là đo đồng thuận giữa hai nguồn, không phải độ chính xác tuyệt đối.
Chạy từ gốc repo:
    python -m src.data_prep.pilot_bbox_check
"""

import json
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "visr_bench_vi"
OUT = ROOT / "data" / "processed" / "visr_bench_vi"
DOC = "0004e63084e48e25a6982f11c7a77115"
PNG_WIDTH = 1653  # mọi trang PNG của tài liệu này rộng 1653 px

# Mỗi mục: (page_id, câu hỏi, đáp án, question_type, evidence_type, ctx hoặc None).
# Đáp án chép nguyên văn từ ảnh trang (đã đối chiếu bằng mắt với ảnh các trang 3, 4, 6, 9).
ITEMS = [
    # Trang 3: bảng thông số kỹ thuật (20 câu)
    (3, "Bo mạch chủ hỗ trợ CPU trong gói nào?", "LGA1156", "table", "table", None),
    (3, "Bộ vi xử lý (chipset) của bo mạch là gì?", "P55 Express", "small_detail", "table", None),
    (3, "Cấu trúc bộ nhớ của bo mạch là gì?", "kênh kép", "text", "table", None),
    (3, "Bộ giải mã âm thanh của bo mạch là gì?", "Realtek ALC889A", "small_detail", "table", None),
    (3, "Bo mạch dùng chip LAN nào?", "RTL8111D", "small_detail", "table", None),
    (3, "Tốc độ mạng LAN của bo mạch là bao nhiêu?", "10/100/1000 Mbit", "small_detail", "table", None),
    (3, "Khe cắm PCI Express nào có tốc độ x4?", "PCIEX4_1", "small_detail", "table", None),
    (3, "Bo mạch hỗ trợ công nghệ đồ họa nào?", "ATI CrossFireX", "diacritic", "table", None),
    (3, "Đầu nối SATA cao nhất trên chip vi xử lý là gì?", "SATA2_5", "small_detail", "table", None),
    (3, "Chip nào được dùng cho đầu nối eSATA?", "JMB362", "small_detail", "table", None),
    (3, "Âm thanh hỗ trợ các kênh nào?", "2/4/5.1/7.1", "small_detail", "table", None),
    (3, "Bo mạch hỗ trợ đầu vào/ra kỹ thuật số nào cho âm thanh?", "S/PDIF", "small_detail", "table", "đầu vào/ra"),
    (3, "Bo mạch hỗ trợ profile bộ nhớ nào?", "Extreme Memory Profile", "text", "table", None),
    (3, "Bo mạch có hỗ trợ mô-đun bộ nhớ loại nào?", "không ECC", "diacritic", "table", None),
    (3, "Thanh nhớ DDR3 được hỗ trợ ở những tốc độ nào?", "2200/1333/1066/800 MHz", "small_detail", "table", None),
    (3, "Các đế cắm DIMM DDR3 có điện áp bao nhiêu?", "1.5V", "small_detail", "table", None),
    (3, "Chip GIGABYTE SATA2 hỗ trợ đầu nối IDE chuẩn nào?", "ATA-133/100/66/33", "small_detail", "table", None),
    (3, "Bộ nhớ cache L3 khác nhau theo đâu?", "từng CPU", "text", "table", None),
    (3, "Bo mạch hỗ trợ công nghệ ATI CrossFireX cùng khe cắm nào?", "PCIEX16_1", "small_detail", "table", "Chỉ các khe cắm"),
    (3, "Khe cắm PCI Express x1 có số lượng bao nhiêu?", "2", "small_detail", "table", "PCI Express x1"),
    # Trang 4: bảng thông số tiếp theo (9 câu)
    (4, "Có đến bao nhiêu cổng USB 2.0/1.1 trên bo mạch?", "14 cổng USB 2.0/1.1", "small_detail", "table", None),
    (4, "Chip nào được dùng cho giao diện IEEE 1394?", "T.I. TSB43AB23", "small_detail", "table", None),
    (4, "Có đến bao nhiêu cổng IEEE 1394a?", "3 cổng IEEE 1394a", "small_detail", "table", None),
    (4, "Đầu nối nguồn điện chính ATX có bao nhiêu chân?", "24 chân", "small_detail", "table", None),
    (4, "Bo mạch có bao nhiêu đầu cắm quạt hệ thống?", "3", "small_detail", "table", "đầu cắm quạt hệ thống"),
    (4, "Bảng mặt sau có bao nhiêu cổng RJ-45?", "2 x cổng RJ-45", "small_detail", "table", None),
    (4, "Bảng mặt sau có bao nhiêu cổng USB 2.0/1.1?", "8 x cổng USB 2.0/1.1", "small_detail", "table", None),
    (4, "Nút nào dùng để xóa sạch CMOS?", "nút xóa sạch CMOS", "diacritic", "table", None),
    (4, "Đầu nối S/PDIF ở bảng mặt sau thuộc loại nào ngoài đồng trục?", "S/PDIF quang học", "diacritic", "table", None),
    # Trang 6: hướng dẫn lắp CPU, có chữ trong hình (4 câu)
    (6, "Cần tra lớp gì lên bề mặt CPU trước khi lắp?", "mỡ chịu nhiệt", "diacritic", "text", None),
    (6, "Không được bật máy tính khi chưa làm gì?", "Không bật máy tính", "text", "text", None),
    (6, "Nhãn nào chỉ chân cắm tam giác trên CPU trong hình?", "Chân cắm tam giác đánh dấu trên CPU", "diacritic", "figure", None),
    (6, "Hình minh họa đế cắm CPU thuộc loại nào?", "Đế cắm CPU LGA1156", "small_detail", "figure", None),
    # Trang 9: lắp bộ nhớ (5 câu)
    (9, "Bo mạch chủ cung cấp bao nhiêu khe cắm bộ nhớ DDR3?", "sáu khe cắm bộ nhớ DDR3", "text", "text", None),
    (9, "Kênh 0 gồm các khe cắm nào?", "DDR3_1, DDR3_2, DDR3_3", "small_detail", "text", None),
    (9, "Kênh 1 gồm các khe cắm nào?", "DDR3_4, DDR3_5, DDR3_6", "small_detail", "text", None),
    (9, "Nếu chỉ lắp một thanh nhớ DDR3 thì nên lắp vào khe nào?", "DDR3_1 hay", "small_detail", "text", "Nếu chỉ lắp một thanh nhớ"),
    (9, "Trong bảng cấu hình kênh kép, SS có nghĩa là gì?", "SS = Một mặt", "diacritic", "table", None),
]


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_marks(s: str) -> str:
    """Bỏ dấu để so khớp OCR; giữ ký tự khác, viết thường."""
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def page_lines(page):
    """Danh sách (văn bản dòng, rect dòng) từ các từ PyMuPDF, nhóm theo (block, line)."""
    groups = defaultdict(list)
    for w in page.get_text("words"):
        groups[(w[5], w[6])].append(w)
    lines = []
    for ws in groups.values():
        ws = sorted(ws, key=lambda w: w[7])
        rect = pymupdf.Rect(min(w[0] for w in ws), min(w[1] for w in ws),
                            max(w[2] for w in ws), max(w[3] for w in ws))
        lines.append((" ".join(w[4] for w in ws), rect))
    return lines


def pdf_bbox(page, lines, answer, ctx, scale):
    """Trả về (số chỗ khớp sau khi lọc ctx, bbox pixel của chỗ khớp đầu tiên hoặc None)."""
    hits = page.search_for(answer)
    if ctx:
        ctx_rects = [r for t, r in lines if ctx in t]
        hits = [h for h in hits if any(r.intersects(h) for r in ctx_rects)]
    if not hits:
        return 0, None
    h = hits[0]
    return len(hits), [round(v * scale, 1) for v in (h.x0, h.y0, h.x1, h.y1)]


MATCH_THRESHOLD = 0.8  # tỉ lệ đoạn khớp dài nhất / độ dài đáp án (sau khi chuẩn hóa)
MAX_WINDOW = 3  # ghép tối đa 3 đoạn OCR liền nhau trên cùng một dòng


def loose(s: str) -> str:
    """Bỏ dấu, chữ thường, chỉ giữ chữ và số: OCR hay tách/đọc sai ký hiệu như '_', '/', '.'."""
    return "".join(c for c in strip_marks(s) if c.isalnum())


def seg_box(seg):
    xs = [p[0] for p in seg[0]]
    ys = [p[1] for p in seg[0]]
    return min(xs), min(ys), max(xs), max(ys)


def ocr_bbox(segs, answer, ctx):
    """Khớp mờ đáp án với 1-3 đoạn OCR liền nhau cùng dòng.

    Trả về (số cửa sổ khớp đạt ngưỡng, bbox đoạn/dòng, bbox từ ước lượng). ctx (nếu có) buộc cửa sổ
    phải chứa cụm ngữ cảnh. Bbox từ ước lượng chia chiều ngang từng đoạn theo vị trí ký tự.
    """
    from difflib import SequenceMatcher

    key = loose(answer)
    ctx_key = loose(ctx) if ctx else None
    texts = [loose(s[1]) for s in segs]
    boxes = [seg_box(s) for s in segs]
    found = []  # (điểm, số đoạn trong cửa sổ, chỉ số đầu, chỉ số cuối, vị trí đầu, độ dài khớp, offsets)
    for j in range(len(segs)):
        for k in range(j, min(j + MAX_WINDOW, len(segs))):
            if abs(boxes[k][1] - boxes[j][1]) > 20:  # khác dòng
                break
            text = "".join(texts[j:k + 1])
            if not text or (ctx_key and ctx_key not in text):
                continue
            m = SequenceMatcher(None, key, text, autojunk=False).find_longest_match(0, len(key), 0, len(text))
            score = m.size / max(len(key), 1)
            if score >= MATCH_THRESHOLD:
                found.append((score, k - j, j, k, m.b, m.size))
    if not found:
        return 0, None, None
    found.sort(key=lambda f: (-f[0], f[1], f[2]))
    score, _, j, k, pos, size = found[0]
    # Đếm số vị trí tách biệt (các cửa sổ chồng nhau trên cùng chỗ chỉ tính một lần)
    chosen = []
    for f in found:
        span = set(range(f[2], f[3] + 1))
        if all(span.isdisjoint(c) for c in chosen):
            chosen.append(span)
    n_places = len(chosen)
    # Các đoạn trong cửa sổ giao với khoảng khớp [pos, pos+size)
    offset, overlapped = 0, []
    for i in range(j, k + 1):
        lo, hi = offset, offset + len(texts[i])
        if hi > pos and lo < pos + size and texts[i]:
            overlapped.append((i, lo))
        offset = hi
    if not overlapped:
        overlapped = [(j, 0)]
    x0 = min(boxes[i][0] for i, _ in overlapped)
    y0 = min(boxes[i][1] for i, _ in overlapped)
    x1 = max(boxes[i][2] for i, _ in overlapped)
    y1 = max(boxes[i][3] for i, _ in overlapped)
    fi, flo = overlapped[0]
    li, llo = overlapped[-1]
    fb, lb = boxes[fi], boxes[li]
    f_frac = max(0.0, (pos - flo) / max(len(texts[fi]), 1))
    l_frac = min(1.0, (pos + size - llo) / max(len(texts[li]), 1))
    word = [round(fb[0] + (fb[2] - fb[0]) * f_frac, 1), int(y0), round(lb[0] + (lb[2] - lb[0]) * l_frac, 1), int(y1)]
    return n_places, [int(x0), int(y0), int(x1), int(y1)], word


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pdf = pymupdf.open(RAW / "src_pdf" / "0004e630" / "doc.pdf")

    import easyocr

    reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)

    pages = sorted({it[0] for it in ITEMS})
    ocr_cache, ocr_time, line_cache = {}, {}, {}
    cache_dir = OUT / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in pages:
        cache_file = cache_dir / f"easyocr_{DOC}_p{p}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            segs, ocr_time[p] = cached["segs"], cached["time_s"]
        else:
            # Đọc bằng PIL rồi truyền mảng: OpenCV không mở được đường dẫn có ký tự tiếng Việt trên Windows
            img = np.array(Image.open(RAW / "hf" / "Multilingual" / DOC / f"{DOC}_{p}.png").convert("RGB"))
            t0 = time.perf_counter()
            raw = reader.readtext(img)
            ocr_time[p] = round(time.perf_counter() - t0, 2)
            segs = [[[[float(x), float(y)] for x, y in poly], text, float(conf)] for poly, text, conf in raw]
            cache_file.write_text(json.dumps({"segs": segs, "time_s": ocr_time[p]}, ensure_ascii=False), encoding="utf-8")
        ocr_cache[p] = (segs, [strip_marks(s[1]) for s in segs])
        line_cache[p] = page_lines(pdf[p])

    rows, times_pdf, word_ious = [], [], []
    counters = defaultdict(int)
    for i, (p, q, a, qt, et, ctx) in enumerate(ITEMS, 1):
        page = pdf[p]
        scale = PNG_WIDTH / page.rect.width
        t1 = time.perf_counter()
        n_pdf, bb_pdf = pdf_bbox(page, line_cache[p], a, ctx, scale)
        times_pdf.append(time.perf_counter() - t1)

        segs, _ = ocr_cache[p]
        n_ocr, bb_line, bb_word = ocr_bbox(segs, a, ctx)
        score = round(iou(bb_pdf, bb_word), 3) if (bb_pdf and bb_word) else None
        if score is not None:
            word_ious.append(score)

        rows.append({
            "qid": f"{DOC}_p{p}_q{i}",
            "document_id": DOC,
            "page_id": p,
            "image": f"hf/Multilingual/{DOC}/{DOC}_{p}.png",
            "question": nfc(q),
            "answer": nfc(a),
            "evidence_type": et,
            "question_type": qt,
            "ctx": ctx,
            "split": "pilot",
            "annotator_model": "claude-haiku-5-5",
            "bbox_pymupdf": bb_pdf,
            "pymupdf_hits": n_pdf,
            "bbox_ocr_line": bb_line,
            "bbox_ocr_word_est": bb_word,
            "ocr_hits": n_ocr,
            "iou_pymupdf_ocr_word_est": score,
        })
        for src, n, bb in (("pymupdf", n_pdf, bb_pdf), ("ocr", n_ocr, bb_line)):
            counters[f"{src}_found"] += bb is not None
            counters[f"{src}_ambiguous"] += n > 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qa_vi_haiku_bbox_pilot.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    by_et = {}
    for et in ("table", "text", "figure"):
        sub = [r for r in rows if r["evidence_type"] == et]
        by_et[et] = {
            "n": len(sub),
            "pymupdf_found": sum(r["bbox_pymupdf"] is not None for r in sub),
            "ocr_found": sum(r["bbox_ocr_line"] is not None for r in sub),
        }
    summary = {
        "n_items": len(rows),
        "pages": pages,
        **dict(counters),
        "both_found": sum(r["bbox_pymupdf"] is not None and r["bbox_ocr_line"] is not None for r in rows),
        "iou_word_est_mean": round(sum(word_ious) / len(word_ious), 3) if word_ious else None,
        "iou_word_est_ge_0_5": sum(v >= 0.5 for v in word_ious),
        "iou_word_est_lt_0_5_qids": [r["qid"] for r in rows
                                     if r["iou_pymupdf_ocr_word_est"] is not None and r["iou_pymupdf_ocr_word_est"] < 0.5],
        "pymupdf_missed": [r["answer"] for r in rows if r["bbox_pymupdf"] is None],
        "ocr_missed": [r["answer"] for r in rows if r["bbox_ocr_line"] is None],
        "by_evidence_type": by_et,
        "time_pymupdf_per_item_s": round(sum(times_pdf) / len(rows), 5),
        "time_ocr_per_page_s": ocr_time,
    }
    (OUT / "bbox_pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
