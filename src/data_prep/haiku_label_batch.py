"""Gán nhãn hàng loạt 25 câu từ bài báo An Bằng và C++ bằng Haiku.

Chạy từ gốc repo:
    python -m src.data_prep.haiku_label_batch
"""

import base64
import json
import os
import sys
import unicodedata
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data" / "raw" / "visr_bench_vi"
OUT_DIR = ROOT / "data" / "processed" / "visr_bench_vi"

SYSTEM_PROMPT = """Bạn là người gán nhãn dữ liệu hỏi đáp tài liệu tiếng Việt.
Dựa vào ảnh trang và text Markdown của trang, hãy tạo một câu hỏi ngắn và đáp án ngắn chép nguyên văn từ trang.
Trả về DUY NHẤT một object JSON:
{
  "question": "câu hỏi ngắn",
  "answer": "đáp án ngắn chép nguyên văn, giữ nguyên dấu tiếng Việt",
  "evidence_type": "text" | "table" | "figure",
  "question_type": "small_detail" | "diacritic" | "table" | "text"
}
Định nghĩa question_type:
- small_detail: đáp án là chi tiết nhỏ (số, mã, ngày, đơn vị, chữ cỡ nhỏ).
- diacritic: đáp án là từ/cụm từ tiếng Việt mà đọc sai dấu thanh sẽ thành sai nghĩa hoặc tên riêng.
- table: phải đọc ô trong bảng.
- text: đọc đoạn văn thường.
Không giải thích, chỉ trả JSON."""

USER_TEMPLATE = """Text trang (tham khảo, có thể lỗi):
<<<
{page_text}
>>>

Tạo một câu hỏi và đáp án từ trang này."""

MAX_PAGE_CHARS = 6000


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def parse_json(content: str) -> dict:
    import re
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        raise ValueError(f"No JSON found in output: {content[:200]!r}")
    return json.loads(m.group(0))


def ask(client: OpenAI, image_url: str, page_text: str) -> dict:
    """Ask Haiku to generate one Q&A pair from the page."""
    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": USER_TEMPLATE.format(page_text=page_text[:MAX_PAGE_CHARS])},
    ]
    resp = client.chat.completions.create(
        model="qwen3-vl-plus",
        temperature=0,
        seed=0,
        max_tokens=1024,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        extra_body={"enable_thinking": False},
    )
    out = parse_json(resp.choices[0].message.content)
    return {
        "question": nfc(str(out.get("question", ""))).strip(),
        "answer": nfc(str(out.get("answer", ""))).strip(),
        "evidence_type": out.get("evidence_type"),
        "question_type": out.get("question_type"),
    }


DOCUMENTS = [
    # An Bằng: 10 trang, tạo ~15 câu (1-2 câu/trang)
    {
        "doc_id": "01d7b45bc59e9652f1764e6fbec2f05e",
        "pages": [0, 1, 2, 3, 4, 5, 6, 7, 8],  # đã gán nhãn p1, thêm p0, p2-p8
        "n_per_page": 1,  # 1-2 câu/trang
    },
    # C++: 3 trang, tạo ~10 câu (2-3 câu/trang)
    {
        "doc_id": "1aa33928936bdf5e0989361a3a68a3a0",
        "pages": [1],  # đã gán nhãn p0, p2; thêm p1 có 1-2 câu
        "n_per_page": 1,
    },
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")

    prov_key = "DASHSCOPE_API_KEY"
    api_key = os.getenv(prov_key)
    if not api_key:
        sys.exit(f"Missing {prov_key} in .env")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    )

    # Load document metadata
    docs = {d["document_id"]: d for d in json.loads((IN_DIR / "documents_vi.json").read_text(encoding="utf-8"))}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "qa_vi_haiku_labels.jsonl"

    # Read existing labels to avoid duplicates
    existing = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.add((row.get("document_id"), row.get("page_id")))

    results = []
    qid_counter = Counter()

    for doc_info in DOCUMENTS:
        doc_id = doc_info["doc_id"]
        doc = docs[doc_id]
        pages = doc_info["pages"]
        n_per = doc_info["n_per_page"]

        for page_id in pages:
            if (doc_id, page_id) in existing:
                print(f"  {doc_id} p{page_id}: already labeled, skip")
                continue

            # Load page image and text
            img_path = IN_DIR / "hf" / "Multilingual" / doc_id / f"{doc_id}_{page_id}.png"
            if not img_path.exists():
                print(f"  {doc_id} p{page_id}: image not found, skip")
                continue

            page_text = doc["page_md"][page_id] if page_id < len(doc["page_md"]) else ""
            img_url = image_data_url(img_path)

            # Generate n_per Q&A pairs for this page
            for i in range(n_per):
                try:
                    result = ask(client, img_url, page_text)
                    qid = f"{doc_id}_{qid_counter[doc_id]}"
                    qid_counter[doc_id] += 1

                    row = {
                        "qid": qid,
                        "document_id": doc_id,
                        "page_id": page_id,
                        "image": f"hf/Multilingual/{doc_id}/{doc_id}_{page_id}.png",
                        "question": result["question"],
                        "answer": result["answer"],
                        "evidence_type": result["evidence_type"],
                        "question_type": result["question_type"],
                        "evidence_bbox": None,
                        "split": "dev",
                        "annotator_model": "qwen3-vl-plus",
                        "note": "Batch-generated by qwen3-vl-plus; check manually before use",
                    }
                    results.append(row)
                    print(f"  {doc_id} p{page_id} q{i+1}: {result['question'][:60]}")
                except Exception as e:
                    print(f"  {doc_id} p{page_id} q{i+1}: ERROR {type(e).__name__}: {e}")

    # Write results
    with open(out_path, "a", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nLabeled {len(results)} items, saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
