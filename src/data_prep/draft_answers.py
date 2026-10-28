"""Sinh đáp án nháp cho câu hỏi VisR-Bench tiếng Việt bằng 2 model (DashScope hoặc OpenRouter).

Đây chỉ là BẢN NHÁP để người duyệt; không dùng trực tiếp làm nhãn dev/test.
- DRAFT_VLM_MODEL: model đọc ảnh, sinh đáp án chính.
- CHECK_MODEL: model trả lời độc lập để kiểm tra chéo; mặc định chỉ đọc text trang.

Lưu ý thiên lệch: hiện chỉ có key Qwen (qwen3-vl-plus), cùng họ với Qwen2.5-VL ở nhánh C2.
Giảm thiểu: chỉ nhận đáp án chép nguyên văn từ trang, model kiểm tra đọc nguồn khác (text
thay vì ảnh), người duyệt 100% dev/test, và ghi rõ trong báo cáo.

Mỗi câu: gửi ảnh trang + text Markdown của trang (trích từ PDF gốc) + câu hỏi, yêu cầu
đáp án ngắn chép nguyên văn từ trang. Sau đó tính:
- `in_page_text`: đáp án (đã chuẩn hóa) có nằm trong text trang không -> span thật.
- `agree`: hai model trả lời giống nhau sau chuẩn hóa; `f1`: token F1 giữa hai đáp án.

Đầu ra: data/processed/visr_bench_vi/qa_vi_draft.jsonl (một dòng một câu, chạy lại sẽ bỏ
qua câu đã xong) và DRAFT_INFO.json (model, prompt, tổng token, chi phí).

Ví dụ:
    python -m src.data_prep.draft_answers --limit 5
    python -m src.data_prep.draft_answers
"""

import argparse
import base64
import json
import os
import re
import sys
import threading
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data" / "raw" / "visr_bench_vi"
OUT_DIR = ROOT / "data" / "processed" / "visr_bench_vi"

SYSTEM_PROMPT = """Bạn là người gán nhãn dữ liệu hỏi đáp tài liệu tiếng Việt.
Bạn nhận một ảnh trang tài liệu, text trích từ trang đó (có thể thiếu hoặc lỗi), và một câu hỏi.
Chỉ dựa vào trang này để trả lời. Trả về DUY NHẤT một object JSON:
{
  "answerable": true/false,      // false nếu trang không chứa đáp án hoặc câu hỏi mơ hồ
  "answer": "...",               // đoạn NGẮN NHẤT chép NGUYÊN VĂN từ trang, giữ nguyên dấu tiếng Việt;
                                 // nếu cần tính toán thì chỉ ghi kết quả; nếu nhiều ý thì nối bằng "; "
                                 // để "" nếu answerable = false
  "evidence_type": "text" | "table" | "figure",   // đáp án nằm trong đoạn văn, bảng, hay hình/biểu đồ
  "question_type": "small_detail" | "diacritic" | "table" | "text"
}
Định nghĩa question_type:
- small_detail: đáp án là chi tiết nhỏ (số, mã, ngày, đơn vị, chữ cỡ nhỏ, chú thích).
- diacritic: đáp án là từ/cụm từ tiếng Việt mà đọc sai dấu thanh sẽ thành sai nghĩa hoặc sai tên riêng.
- table: phải đọc ô trong bảng để trả lời.
- text: đọc đoạn văn thường.
Không giải thích, không thêm chữ nào ngoài JSON."""

USER_TEMPLATE = """Text trích từ trang (tham khảo, có thể lỗi):
<<<
{page_text}
>>>

Câu hỏi: {question}"""

# Giới hạn text trang để tránh prompt quá dài với vài trang dày đặc
MAX_PAGE_CHARS = 6000


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def norm_answer(text: str) -> str:
    """Chuẩn hóa để so khớp: NFC, chữ thường, gộp khoảng trắng, bỏ dấu câu ở hai đầu. Giữ dấu thanh."""
    text = nfc(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,;:!?\"'“”()[]")


def token_f1(a: str, b: str) -> float:
    ta, tb = norm_answer(a).split(), norm_answer(b).split()
    if not ta or not tb:
        return float(ta == tb)
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def parse_json(content: str) -> dict:
    # Một số model bọc JSON trong ```json ... ```
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        raise ValueError(f"Không thấy JSON trong output: {content[:200]!r}")
    return json.loads(m.group(0))


def image_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


PROVIDERS = {
    # extra_body riêng từng nhà cung cấp
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "url_env": "OPENROUTER_BASE_URL",
        "default_url": "https://openrouter.ai/api/v1",
        # Trả về chi phí thực trong usage; giảm suy luận để tiết kiệm token
        "extra_body": {"usage": {"include": True}, "reasoning": {"effort": "low"}},
    },
    "dashscope": {
        "key_env": "DASHSCOPE_API_KEY",
        "url_env": "DASHSCOPE_BASE_URL",
        "default_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        # Tắt chế độ thinking của Qwen3 (DashScope không trả chi phí, chỉ trả token)
        "extra_body": {"enable_thinking": False},
    },
}


def ask(client: OpenAI, model: str, extra_body: dict, image_url: str | None, page_text: str, question: str) -> dict:
    """image_url = None: gửi chỉ text (dùng cho model kiểm tra chéo không đọc ảnh)."""
    content = [{"type": "text", "text": USER_TEMPLATE.format(page_text=page_text[:MAX_PAGE_CHARS], question=question)}]
    if image_url:
        content.insert(0, {"type": "image_url", "image_url": {"url": image_url}})
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=0,
        max_tokens=2048,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
        extra_body=extra_body,
    )
    out = parse_json(resp.choices[0].message.content)
    usage = resp.usage.model_dump() if resp.usage else {}
    return {
        "answerable": bool(out.get("answerable")),
        "answer": nfc(str(out.get("answer", ""))).strip(),
        "evidence_type": out.get("evidence_type"),
        "question_type": out.get("question_type"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Sinh đáp án nháp bằng 2 model qua API tương thích OpenAI.")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N câu đầu (0 = tất cả).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--provider", choices=list(PROVIDERS), default=os.getenv("DRAFT_PROVIDER", "dashscope"))
    parser.add_argument("--draft-model", default=os.getenv("DRAFT_VLM_MODEL"))
    parser.add_argument("--check-model", default=os.getenv("CHECK_MODEL"))
    parser.add_argument(
        "--check-text-only",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("CHECK_TEXT_ONLY", "1") == "1",
        help="Model kiểm tra chỉ đọc text trang, không đọc ảnh (mặc định: có).",
    )
    args = parser.parse_args()

    prov = PROVIDERS[args.provider]
    api_key = os.getenv(prov["key_env"])
    if not api_key:
        sys.exit(f"Thiếu {prov['key_env']} trong .env")
    if not args.draft_model or not args.check_model:
        sys.exit("Thiếu DRAFT_VLM_MODEL / CHECK_MODEL trong .env (hoặc --draft-model / --check-model)")
    client = OpenAI(api_key=api_key, base_url=os.getenv(prov["url_env"], prov["default_url"]))

    docs = {d["document_id"]: d for d in json.loads((IN_DIR / "documents_vi.json").read_text(encoding="utf-8"))}
    questions = [json.loads(line) for line in (IN_DIR / "qa_vi.jsonl").read_text(encoding="utf-8").splitlines()]
    # qid ổn định: document_id + thứ tự câu trong tài liệu
    seen = Counter()
    for q in questions:
        q["qid"] = f"{q['document_id']}_{seen[q['document_id']]}"
        seen[q["document_id"]] += 1
    if args.limit:
        questions = questions[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "qa_vi_draft.jsonl"
    done = set()
    if out_path.exists():
        # Giữ dòng đã thành công; dòng lỗi bị bỏ để chạy lại (tránh trùng qid trong file)
        old = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        keep = [r for r in old if "error" not in r["draft"] and "error" not in r["check"]]
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
        done = {r["qid"] for r in keep}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"{len(questions)} câu, đã xong {len(questions) - len(todo)}, cần chạy {len(todo)}")
    print(f"[{args.provider}] model nháp: {args.draft_model} | model kiểm tra: {args.check_model}"
          f"{' (chỉ text)' if args.check_text_only else ''}")

    lock = threading.Lock()

    def process(q: dict) -> dict:
        page_text = docs[q["document_id"]]["page_md"][q["page_id"]]
        img = image_data_url(IN_DIR / q["image"])
        row = dict(q)
        check_img = None if args.check_text_only else img
        for key, model, image in (("draft", args.draft_model, img), ("check", args.check_model, check_img)):
            try:
                row[key] = ask(client, model, prov["extra_body"], image, page_text, q["question"])
            except Exception as e:  # lưu lỗi để chạy lại sau, không dừng cả lô
                row[key] = {"error": f"{type(e).__name__}: {e}"[:500]}
        a, b = row["draft"].get("answer", ""), row["check"].get("answer", "")
        page_norm = norm_answer(page_text)
        row["in_page_text"] = bool(a) and norm_answer(a) in page_norm
        row["agree"] = bool(a) and norm_answer(a) == norm_answer(b)
        row["f1"] = round(token_f1(a, b), 3)
        return row

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(out_path, "a", encoding="utf-8") as f:
        futures = [pool.submit(process, q) for q in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            has_err = "error" in row["draft"] or "error" in row["check"]
            n_err += has_err
            n_ok += not has_err
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} (lỗi: {n_err})")

    # Tổng hợp trên toàn bộ file (kể cả các lần chạy trước)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [r for r in rows if "error" not in r["draft"] and "error" not in r["check"]]
    cost = {k: sum((r[k].get("cost") or 0) for r in rows) for k in ("draft", "check")}
    tokens = {k: sum((r[k].get("prompt_tokens") or 0) + (r[k].get("completion_tokens") or 0) for r in rows)
              for k in ("draft", "check")}
    summary = {
        "n_rows": len(rows),
        "n_ok": len(ok),
        "answerable_draft": sum(r["draft"]["answerable"] for r in ok),
        "agree": sum(r["agree"] for r in ok),
        "in_page_text": sum(r["in_page_text"] for r in ok),
        "mean_f1": round(sum(r["f1"] for r in ok) / max(len(ok), 1), 3),
        "evidence_type": Counter(r["draft"]["evidence_type"] for r in ok),
        "question_type": Counter(r["draft"]["question_type"] for r in ok),
        "cost_usd": cost,
        "tokens": tokens,
    }
    info = {
        "provider": args.provider,
        "draft_model": args.draft_model,
        "check_model": args.check_model,
        "check_text_only": args.check_text_only,
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "max_page_chars": MAX_PAGE_CHARS,
        "params": {"temperature": 0, "seed": 0, "max_tokens": 2048, "extra_body": prov["extra_body"]},
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
    }
    (OUT_DIR / "DRAFT_INFO.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
