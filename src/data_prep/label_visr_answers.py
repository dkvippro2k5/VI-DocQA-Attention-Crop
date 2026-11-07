"""Gán đáp án cho câu hỏi tiếng Việt có sẵn của VisR-Bench (mọi tài liệu đã tải trong documents_vi.json).

Câu hỏi giữ nguyên từ multilingual_QA.json (đáp án gốc để trống); Qwen3-VL đọc ảnh trang
và chép đáp án ngắn. Dùng endpoint OpenAI-compatible quốc tế trong .env (cùng haiku_label_batch.py).

Chạy từ gốc repo:
    python -X utf8 -m src.data_prep.label_visr_answers --dry-run     # chỉ đếm, không gọi API
    python -X utf8 -m src.data_prep.label_visr_answers --limit 100   # gán tối đa 100 câu chưa có
Chạy lại sẽ tiếp tục từ chỗ dừng (bỏ qua qid đã có trong file đầu ra).
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data" / "raw" / "visr_bench_vi"
OUT_PATH = ROOT / "data" / "processed" / "visr_bench_vi" / "qa_vi_visr_answers.jsonl"
MODEL = "qwen3-vl-plus"
MAX_PAGE_CHARS = 6000

SYSTEM_PROMPT = """Bạn là người gán đáp án cho bộ dữ liệu hỏi đáp tài liệu tiếng Việt.
Cho ảnh một trang, text Markdown của trang và một câu hỏi. Hãy trả lời bằng đáp án NGẮN NHẤT,
chép nguyên văn từ trang, giữ nguyên dấu tiếng Việt. Nếu trang không chứa đáp án, để answer rỗng.
Trả về DUY NHẤT một object JSON:
{
  "answer": "đáp án ngắn chép nguyên văn, hoặc rỗng nếu trang không có",
  "evidence_type": "text" | "table" | "figure",
  "question_type": "small_detail" | "diacritic" | "table" | "text"
}
question_type:
- small_detail: đáp án là chi tiết nhỏ (số, mã, ngày, đơn vị, chữ cỡ nhỏ).
- diacritic: đáp án là từ/cụm từ tiếng Việt mà đọc sai dấu thanh sẽ sai nghĩa hoặc tên riêng.
- table: phải đọc ô trong bảng.
- text: đọc đoạn văn thường.
Không giải thích, chỉ trả JSON."""

USER_TEMPLATE = """Text trang (tham khảo, có thể lỗi):
<<<
{page_text}
>>>

Câu hỏi: {question}"""


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


EXCLUDED_FILE = ROOT / "data" / "qa" / "excluded_documents.json"


def excluded_docs() -> set:
    """document_id bị loại khỏi bộ dữ liệu (data/qa/excluded_documents.json)."""
    if not EXCLUDED_FILE.exists():
        return set()
    return {d["document_id"] for d in json.loads(EXCLUDED_FILE.read_text(encoding="utf-8"))["documents"]}


GARBLED = set("˛˝˜ˆ˙°ˇ˘˚¸�")


def pdf_page_texts(doc_id: str, n_pages: int):
    """Text từng trang lấy từ PDF gốc (src_pdf/<doc[:8]>/doc.pdf) nếu có và đọc được; None nếu không dùng được.

    page_md của VisR-Bench ở nhiều tài liệu bị mất dấu hoặc lỗi font ('TR˛˝NG TIU HC'), trong khi lớp text PDF
    vẫn đúng. Trang PDF scan (gần như không có chữ) hoặc lỗi font thì trả None cho trang đó."""
    pdf = IN_DIR / "src_pdf" / doc_id[:8] / "doc.pdf"
    if not pdf.exists():
        return None
    import pymupdf
    with pymupdf.open(pdf) as doc:
        if len(doc) != n_pages:
            return None
        out = []
        for page in doc:
            t = nfc(page.get_text())
            bad = sum(c in GARBLED for c in t) / max(1, len(t))
            out.append(t if len(t.split()) >= 5 and bad < 0.005 else None)
    return out


def collect_questions() -> list:
    docs = {d["document_id"]: d for d in json.loads((IN_DIR / "documents_vi.json").read_text(encoding="utf-8"))}
    skip = excluded_docs()
    items = []
    for d in json.loads((IN_DIR / "multilingual_QA.json").read_text(encoding="utf-8")):
        doc_id = d["file_name"]
        if doc_id not in docs or doc_id in skip:
            continue
        md = docs[doc_id]["page_md"]
        pdf_txt = pdf_page_texts(doc_id, len(md)) or [None] * len(md)
        for idx, qa in enumerate(d["qa_list"]):
            if qa.get("detected_language") != "vi" or not qa.get("question"):
                continue
            page = qa.get("page_index", 0)
            img = f"hf/Multilingual/{doc_id}/{doc_id}_{page}.png"
            items.append({
                "qid": f"{doc_id}_p{page}_v{idx}",
                "document_id": doc_id,
                "page_id": page,
                "image": img,
                "question": nfc(qa["question"]).strip(),
                "page_text": (pdf_txt[page] or md[page]) if page < len(md) else "",
            })
    return items


def ask(client, item: dict) -> dict:
    img_path = IN_DIR / item["image"]
    url = "data:image/png;base64," + base64.b64encode(img_path.read_bytes()).decode()
    content = [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": USER_TEMPLATE.format(page_text=item["page_text"][:MAX_PAGE_CHARS],
                                                      question=item["question"])},
    ]
    last = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0, seed=0, max_tokens=512, timeout=120,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                extra_body={"enable_thinking": False},
            )
            m = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
            if not m:
                raise ValueError("no JSON in output")
            out = json.loads(m.group(0))
            return {"answer": nfc(str(out.get("answer", ""))).strip(),
                    "evidence_type": out.get("evidence_type"),
                    "question_type": out.get("question_type")}
        except Exception as e:  # lỗi mạng/JSON: thử lại với backoff
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{type(last).__name__}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    items = collect_questions()
    done = set()
    if OUT_PATH.exists():
        done = {json.loads(l)["qid"] for l in OUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [i for i in items if i["qid"] not in done]

    per_doc = {}
    for i in items:
        per_doc[i["document_id"]] = per_doc.get(i["document_id"], 0) + 1
    missing = [i["image"] for i in items if not (IN_DIR / i["image"]).exists()]
    print(f"Tổng câu hỏi: {len(items)} | đã có: {len(done)} | còn: {len(todo)} | thiếu ảnh: {len(missing)}")
    print("Theo tài liệu:", {k[:8]: v for k, v in per_doc.items()})
    if args.dry_run:
        return

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        sys.exit("Thiếu DASHSCOPE_API_KEY trong .env")
    client = OpenAI(api_key=key, base_url=os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for n, item in enumerate(todo[: args.limit], 1):
            try:
                res = ask(client, item)
            except Exception as e:
                fail += 1
                print(f"[{n}] {item['qid'][-18:]} LỖI {e}")
                continue
            row = {k: item[k] for k in ("qid", "document_id", "page_id", "image", "question")}
            row.update(res)
            row.update({"evidence_bbox": None, "split": "", "annotator_model": MODEL,
                        "note": "Câu hỏi gốc VisR-Bench; đáp án do VLM chép từ ảnh, cần kiểm tra"})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            ok += 1
            print(f"[{n}] {item['qid'][-18:]} | {item['question'][:40]} -> {res['answer'][:40]!r}")
    print(f"\nXong: {ok} câu, lỗi {fail}. Ghi vào {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
