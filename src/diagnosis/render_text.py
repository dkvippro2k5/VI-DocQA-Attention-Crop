"""Sinh bộ ảnh chữ chẩn đoán head cho HAVC: tiếng Anh (C4) và tiếng Việt có dấu (C5).

HAVC (Xie và cs., arXiv:2601.22483) chẩn đoán head bằng 1.000 ảnh OCR Synthdog nhưng không mô tả cách vẽ ảnh
hay vùng chữ dùng làm mặt nạ. Ở đây tự vẽ ảnh kiểu Synthdog đơn giản (khối chữ trên nền giấy) và lưu khung của
từng từ để làm mặt nạ m_i. Khác paper: không dùng ảnh Synthdog gốc vì ảnh đó không kèm khung chữ.

Kiểm soát biến (scope mục 6): ảnh thứ i của hai bộ dùng chung bố cục (font, cỡ chữ, kích thước ảnh, nền, màu chữ,
lề, giãn dòng, số dòng tối đa); chỉ khác văn bản. Cả hai bộ dùng cùng 8 font mã nguồn mở có đủ dấu tiếng Việt.

Văn bản: đoạn văn Wikipedia (wikimedia/wikipedia 20231101, đọc streaming). Tiếng Anh chỉ giữ đoạn toàn ASCII
(chữ Latin không dấu như HAVC gốc); tiếng Việt chỉ giữ đoạn có nhiều từ mang dấu. Bỏ bài có từ khóa chính trị,
tôn giáo, chiến tranh, tình dục, ma túy, bạo lực (ngưỡng nội dung của đồ án). Không dùng tài liệu dev/test.

Ra: data/diagnostic/<lang>/<id>.png và data/diagnostic/<lang>/meta.jsonl, mỗi dòng
  {"id", "image", "size": [W, H], "text": các dòng nối bằng "\\n", "words": [[từ, x0, y0, x1, y1, số_dòng], ...],
   "layout": {...}, "source": tên bài}
Văn bản đã chọn được lưu đệm ở data/diagnostic/corpus_<lang>.jsonl để chạy lại cho cùng kết quả.

Chạy từ gốc repo:
    python -X utf8 -m src.diagnosis.render_text --n 1000
"""

import argparse
import json
import random
import re
import unicodedata
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "diagnostic"
FONT_DIR = OUT / "fonts"
SEED = 0
WIKI = {"en": "20231101.en", "vi": "20231101.vi"}
FONT_URLS = {  # google/fonts, giấy phép OFL; đều có đủ ký tự tiếng Việt
    "NotoSans": "ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf",
    "NotoSerif": "ofl/notoserif/NotoSerif%5Bwdth,wght%5D.ttf",
    "Roboto": "ofl/roboto/Roboto%5Bwdth,wght%5D.ttf",
    "BeVietnamPro": "ofl/bevietnampro/BeVietnamPro-Regular.ttf",
    "OpenSans": "ofl/opensans/OpenSans%5Bwdth,wght%5D.ttf",
    "Merriweather": "ofl/merriweather/Merriweather%5Bopsz,wdth,wght%5D.ttf",
    "Lora": "ofl/lora/Lora%5Bwght%5D.ttf",
    "SourceSans3": "ofl/sourcesans3/SourceSans3%5Bwght%5D.ttf",
}
# so khớp trên chữ thường đã bỏ dấu, theo ranh giới từ
SENSITIVE = {
    "vi": ["chinh tri", "dang cong san", "cong san", "tong bi thu", "chu tich nuoc", "chu tich ho chi minh",
           "ho chi minh", "quoc hoi", "chinh phu", "ton giao", "phat giao", "cong giao", "thien chua", "giao hoi",
           "nha tho", "chua giesu", "hoi giao", "tin lanh", "chien tranh", "quan doi", "khang chien", "bieu tinh",
           "khung bo", "cach mang", "viet nam cong hoa", "dao phat", "tu vien",
           "mai dam", "tinh duc", "khieu dam", "ma tuy", "giet nguoi", "tu tu", "hiep dam"],
    "en": ["politic", "communis", "socialis", "anarch", "religio", "church", "christian", "catholic", "islam",
           "muslim", "buddhis", "hindu", "jewish", "judaism", "bible", "jesus", "god", "gods", "war", "wars",
           "military", "army", "terror", "nazi", "election", "government", "parliament", "party", "revolution",
           "genocide", "holocaust", "sex", "porn", "prostitut", "drug", "murder", "suicide", "rape"],
}
VI_MARK = re.compile(r"[̀-̣̉]|đ", re.IGNORECASE)


def no_accent(s: str) -> str:
    return unicodedata.normalize("NFD", s.lower()).replace("đ", "d").encode("ascii", "ignore").decode()


def has_mark(word: str) -> bool:
    """Từ tiếng Việt có dấu phụ (thanh hoặc mũ/móc/ngang): ă â ê ô ơ ư đ và 5 dấu thanh."""
    return bool(VI_MARK.search(unicodedata.normalize("NFD", word)))


def sensitive(text: str, lang: str) -> bool:
    t = " " + re.sub(r"[^a-z0-9]+", " ", no_accent(text)) + " "
    return any(f" {k}" in t if lang == "en" else f" {k} " in t for k in SENSITIVE[lang])


def ok_paragraph(p: str, lang: str) -> bool:
    words = p.split()
    if len(words) < 45 or re.search(r"[|{}=<>\[\]#*_\\]|https?:", p):
        return False
    if lang == "en":
        return p.isascii() and sum(w.isalpha() for w in words) / len(words) > 0.8
    marked = sum(has_mark(w) for w in words)
    return marked / len(words) > 0.4 and all(ord(c) < 0x2000 or c in "–—‘’“”…" for c in p)


def build_corpus(lang: str, n: int) -> list:
    """Mỗi bài lấy tối đa một đoạn, để văn bản đa dạng chủ đề."""
    cache = OUT / f"corpus_{lang}.jsonl"
    if cache.exists():
        rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows) >= n:
            return rows[:n]
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", WIKI[lang], split="train", streaming=True)
    rows, seen, skipped = [], 0, 0
    for art in ds:
        seen += 1
        head = art["title"] + " " + art["text"][:3000]
        if sensitive(head, lang):
            skipped += 1
            continue
        for p in art["text"].split("\n"):
            p = unicodedata.normalize("NFC", re.sub(r"\s+", " ", p).strip())
            if ok_paragraph(p, lang) and not sensitive(p, lang):
                rows.append({"title": art["title"], "text": p})
                break
        if len(rows) >= n:
            break
    print(f"  {lang}: đọc {seen} bài, bỏ {skipped} bài nhạy cảm, lấy {len(rows)} đoạn")
    cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def fonts() -> dict:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, rel in FONT_URLS.items():
        p = FONT_DIR / f"{name}.ttf"
        if not p.exists():
            urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/" + rel, p)
        out[name] = p
    return out


def layout(i: int, font_names: list) -> dict:
    """Bố cục của ảnh thứ i, chỉ phụ thuộc i (dùng chung cho hai ngôn ngữ)."""
    r = random.Random(SEED * 100003 + i)
    W, H = r.randrange(512, 833, 4), r.randrange(384, 641, 4)
    paper = r.randint(225, 255)
    return {
        "W": W, "H": H, "font": r.choice(font_names), "size": r.randint(18, 30),
        "bg": [paper, paper - r.randint(0, 12), paper - r.randint(0, 25)],
        "ink": [r.randint(0, 60)] * 3, "margin": r.randint(16, 48), "spacing": round(r.uniform(1.25, 1.7), 2),
        "max_lines": r.randint(3, 7), "noise": round(r.uniform(0, 6), 1), "blur": r.random() < 0.3,
    }


def render(text: str, lay: dict, font_path: Path, seed: int):
    W, H = lay["W"], lay["H"]
    img = Image.new("RGB", (W, H), tuple(lay["bg"]))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_path), lay["size"])
    line_h = round(lay["size"] * lay["spacing"])
    x0, x1 = lay["margin"], W - lay["margin"]
    n_lines = min(lay["max_lines"], (H - 2 * lay["margin"]) // line_h)
    space = draw.textlength(" ", font=font)
    words, lines, cur, x, li = [], [], [], x0, 0
    y = lay["margin"]
    for w in text.split():
        wl = draw.textlength(w, font=font)
        if cur and x + wl > x1:  # xuống dòng
            lines.append(" ".join(cur)); cur, x, li, y = [], x0, li + 1, y + line_h
        if li >= n_lines or wl > x1 - x0:
            break
        draw.text((x, y), w, font=font, fill=tuple(lay["ink"]), anchor="la")
        bx = draw.textbbox((x, y), w, font=font, anchor="la")  # khung mực thật, gồm cả dấu trên/dưới
        words.append([w, *bx, li])
        cur.append(w); x += wl + space
    if cur and li < n_lines:
        lines.append(" ".join(cur))
    if lay["noise"]:
        import numpy as np
        rng = np.random.default_rng(seed)
        arr = np.asarray(img, dtype=np.float32) + rng.normal(0, lay["noise"], (H, W, 1))
        img = Image.fromarray(arr.clip(0, 255).astype("uint8"))
    if lay["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img, "\n".join(lines), words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--langs", nargs="+", default=["en", "vi"])
    args = ap.parse_args()
    fp = fonts()
    names = sorted(fp)
    for lang in args.langs:
        corpus = build_corpus(lang, args.n)
        d = OUT / lang
        d.mkdir(parents=True, exist_ok=True)
        meta = []
        for i in range(args.n):
            lay = layout(i, names)
            img, text, words = render(corpus[i]["text"], lay, fp[lay["font"]], seed=i)
            name = f"{lang}_{i:04d}.png"
            img.save(d / name)
            meta.append({"id": f"{lang}_{i:04d}", "image": f"{lang}/{name}", "size": [lay["W"], lay["H"]],
                         "text": text, "words": words, "layout": lay, "source": corpus[i]["title"]})
        (d / "meta.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta),
                                      encoding="utf-8")
        nw = [len(m["words"]) for m in meta]
        nm = sum(has_mark(w[0]) for m in meta for w in m["words"])
        print(f"{lang}: {len(meta)} ảnh, {sum(nw)} từ (trung bình {sum(nw) / len(nw):.1f}/ảnh), "
              f"từ có dấu {nm / max(1, sum(nw)):.0%}")


if __name__ == "__main__":
    main()
