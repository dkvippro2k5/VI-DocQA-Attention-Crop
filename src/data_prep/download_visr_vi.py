"""Tải mẫu phần tiếng Việt của VisR-Bench (arXiv 2508.07493).

VisR-Bench gồm 373 tài liệu tiếng Anh + 913 tài liệu đa ngữ; phần tiếng Việt có 27 PDF
(1.934 câu hỏi). Câu hỏi do GPT-4o sinh, file QA công khai KHÔNG có đáp án, chỉ có
`page_index` (trang chứa minh chứng). Dùng làm mẫu thử quy trình gán nhãn, không làm test chính.

- QA + text Markdown từng trang: GitHub puar-playground/VisR-Bench, QA/multilingual_QA.json
- Ảnh trang (PNG): HF dataset puar-playground/VisR-Bench, thư mục Multilingual/<file_name>/

Đầu ra (data/raw/visr_bench_vi/):
    docs/<file_name>/<file_name>_<i>.png   ảnh từng trang
    qa_vi.jsonl                            mỗi dòng một câu hỏi (document_id, page_id, question, ...)
    documents_vi.json                      metadata tài liệu (url, số trang, text từng trang)
    DOWNLOAD_INFO.json

Ví dụ (chạy từ gốc repo):
    python -m src.data_prep.download_visr_vi                 # 5 tài liệu ngẫu nhiên, <= 60 trang
    python -m src.data_prep.download_visr_vi --n-docs 0      # cả 27 tài liệu
"""

import argparse
import json
import random
import sys
import unicodedata
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

HF_REPO_ID = "puar-playground/VisR-Bench"
GITHUB_REPO = "puar-playground/VisR-Bench"
# Commit cuối của repo GitHub tại thời điểm viết script (2025-08-12)
GITHUB_SHA = "12df015fa64224ff60bc1d810c41d7cb139a7f0b"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "visr_bench_vi"


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def doc_language(doc: dict) -> str:
    """Ngôn ngữ của tài liệu = ngôn ngữ xuất hiện nhiều nhất trong các câu hỏi của nó."""
    return Counter(q["detected_language"] for q in doc["qa_list"]).most_common(1)[0][0]


def load_multilingual_qa(cache_path: Path) -> list[dict]:
    if not cache_path.exists():
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_SHA}/QA/multilingual_QA.json"
        print(f"Tải {url} (~60 MB)")
        urllib.request.urlretrieve(url, cache_path)
    return json.loads(cache_path.read_text(encoding="utf-8"))


def main() -> None:
    # Console Windows mặc định cp1252, không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Tải mẫu phần tiếng Việt của VisR-Bench.")
    parser.add_argument("--n-docs", type=int, default=5, help="Số tài liệu cần tải; 0 = tất cả.")
    parser.add_argument("--max-pages", type=int, default=60, help="Bỏ tài liệu dài hơn (0 = không giới hạn).")
    parser.add_argument("--doc-ids", nargs="+", help="Chỉ định file_name cụ thể (bỏ qua --n-docs).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_docs = load_multilingual_qa(args.out_dir / "multilingual_QA.json")
    vi_docs = sorted((d for d in all_docs if doc_language(d) == "vi"), key=lambda d: d["file_name"])
    print(f"Tìm thấy {len(vi_docs)} tài liệu tiếng Việt")

    if args.doc_ids:
        wanted = set(args.doc_ids)
        chosen = [d for d in vi_docs if d["file_name"] in wanted]
        missing = wanted - {d["file_name"] for d in chosen}
        if missing:
            parser.error(f"Không phải tài liệu tiếng Việt hoặc không tồn tại: {sorted(missing)}")
    else:
        pool = [d for d in vi_docs if not args.max_pages or len(d["all_page_images"]) <= args.max_pages]
        n = len(pool) if args.n_docs == 0 else min(args.n_docs, len(pool))
        chosen = sorted(random.Random(args.seed).sample(pool, n), key=lambda d: d["file_name"])

    hf_sha = HfApi().dataset_info(HF_REPO_ID).sha
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=hf_sha,
        local_dir=args.out_dir / "hf",
        allow_patterns=[f"Multilingual/{d['file_name']}/*.png" for d in chosen],
    )

    qa_rows, doc_rows = [], []
    for d in chosen:
        doc_id = d["file_name"]
        for q in d["qa_list"]:
            qa_rows.append({
                "document_id": doc_id,
                "page_id": q["page_index"],
                "image": f"hf/Multilingual/{doc_id}/{d['all_page_images'][q['page_index']]}",
                "question": nfc(q["question"]),
                # Bản công khai không có đáp án; giữ nguyên chuỗi rỗng để biết cần gán
                "answer": nfc(q["answer"]),
                "detected_language": q["detected_language"],
            })
        doc_rows.append({
            "document_id": doc_id,
            "url": d["url"],
            "n_pages": len(d["all_page_images"]),
            "n_questions": len(d["qa_list"]),
            "page_md": [nfc(p) for p in d["all_page_md_str"]],
        })

    with open(args.out_dir / "qa_vi.jsonl", "w", encoding="utf-8") as f:
        for row in qa_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "documents_vi.json").write_text(
        json.dumps(doc_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    info = {
        "hf_repo_id": HF_REPO_ID,
        "hf_revision": hf_sha,
        "github_repo": GITHUB_REPO,
        "github_revision": GITHUB_SHA,
        "document_ids": [d["file_name"] for d in chosen],
        "seed": args.seed,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out_dir / "DOWNLOAD_INFO.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for d in doc_rows:
        print(f"  {d['document_id']}  {d['n_pages']:>3} trang  {d['n_questions']:>3} câu  {d['url']}")
    print(f"Xong: {len(doc_rows)} tài liệu, {len(qa_rows)} câu hỏi -> {args.out_dir}")


if __name__ == "__main__":
    main()
