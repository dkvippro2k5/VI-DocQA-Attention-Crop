"""Gán nhãn toàn bộ 1934 QA pairs tiếng Việt từ VisR-Bench VI.

Dùng Qwen3-VL qua DashScope API. Gán batch 100 items, lưu checkpoint, dừng để kiểm tra.

Chạy từ gốc repo:
    python -m src.data_prep.label_visr_vi_full

Hoặc trên Colab với DashScope API key.
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import Optional
import unicodedata

import requests
from PIL import Image
import io
import base64

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data" / "raw" / "visr_bench_vi"
OUT_DIR = ROOT / "data" / "processed" / "visr_bench_vi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DashScope API
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/api/v1/services/multimodal/text-image-to-text"
MODEL = "qwen3-vl-plus"

BATCH_SIZE = 100
MAX_ITEMS = 100  # Gán 100 cái đầu để kiểm tra


def norm_text(s: str) -> str:
    """Chuẩn hóa text: NFC, lowercase, trim whitespace."""
    return " ".join(unicodedata.normalize("NFC", s).lower().split())


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_qa_with_qwen(image_base64: str, page_text: str, retries: int = 3) -> Optional[dict]:
    """Gọi Qwen3-VL để sinh câu hỏi và đáp án từ ảnh và text trang."""

    if not DASHSCOPE_API_KEY:
        print("ERROR: DASHSCOPE_API_KEY not set")
        return None

    prompt = f"""Dựa vào hình ảnh trang và nội dung text dưới đây, hãy:
1. Sinh một câu hỏi tiếng Việt ngắn gọn (5-15 từ) về nội dung trang
2. Trả lời câu hỏi dựa trên text (1-10 từ)

Text trang:
{page_text[:500]}

Format trả lời JSON:
{{"question": "...", "answer": "...", "question_type": "small_detail|diacritic|table|text"}}
"""

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image/png;base64,{image_base64}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        },
        "parameters": {
            "enable_thinking": False,
        },
    }

    for attempt in range(retries):
        try:
            resp = requests.post(DASHSCOPE_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            if "output" in result and "choices" in result["output"]:
                text = result["output"]["choices"][0]["message"]["content"][0]["text"]
                # Parse JSON from response
                try:
                    qa = json.loads(text)
                    return qa
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        qa = json.loads(text[start:end])
                        return qa
                    print(f"  Failed to parse JSON from response: {text[:100]}")
                    return None
            else:
                print(f"  Unexpected response format: {result}")
                return None
        except Exception as e:
            print(f"  Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    # Load multilingual QA
    with open(IN_DIR / "multilingual_QA.json", encoding="utf-8") as f:
        all_docs = json.load(f)

    # Load documents metadata
    with open(IN_DIR / "documents_vi.json", encoding="utf-8") as f:
        docs_by_id = {doc["document_id"]: doc for doc in json.load(f)}

    print(f"Loaded {len(all_docs)} documents from multilingual_QA.json")
    print(f"Loaded {len(docs_by_id)} document metadata")

    # Extract VI QA pairs
    vi_items = []
    for doc in all_docs:
        doc_id = doc["file_name"]
        for qa_idx, qa in enumerate(doc["qa_list"]):
            if qa.get("detected_language") == "vi" and qa.get("question") and qa.get("answer"):
                vi_items.append({
                    "document_id": doc_id,
                    "page_id": qa.get("page_index", 0),
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "page_text": doc["all_page_md_str"][qa.get("page_index", 0)] if doc["all_page_md_str"] else "",
                    "image_data": doc["all_page_images"][qa.get("page_index", 0)] if doc["all_page_images"] else None,
                })

    print(f"Extracted {len(vi_items)} VI QA pairs")

    # Limit to MAX_ITEMS for checkpoint
    vi_items = vi_items[:MAX_ITEMS]
    print(f"Processing {len(vi_items)} items for checkpoint")

    # Generate labels
    labeled = []
    for idx, item in enumerate(vi_items):
        qid = f"visr_vi_{idx:05d}"
        doc_id = item["document_id"]
        page_id = item["page_id"]

        print(f"\n[{idx+1}/{len(vi_items)}] {qid}")
        print(f"  Document: {doc_id}, Page: {page_id}")
        print(f"  Question (original): {item['question'][:60]}...")
        print(f"  Answer (original): {item['answer']}")

        # Find image path
        img_path = IN_DIR / "hf" / "Multilingual" / doc_id / f"{doc_id}_{page_id}.png"
        if not img_path.exists():
            print(f"  Image not found at {img_path}, skip")
            continue

        # Load and encode image
        try:
            img = Image.open(img_path).convert("RGB")
            img_base64 = image_to_base64(img)
        except Exception as e:
            print(f"  Failed to load/encode image: {e}")
            continue

        # Generate question and answer with Qwen
        qa_result = generate_qa_with_qwen(img_base64, item["page_text"])
        if not qa_result:
            print(f"  Qwen generation failed, use original")
            qa_result = {
                "question": item["question"],
                "answer": item["answer"],
                "question_type": "text",
            }

        # Create label record
        label = {
            "qid": qid,
            "document_id": doc_id,
            "page_id": page_id,
            "image": str(Path("hf") / "Multilingual" / doc_id / f"{doc_id}_{page_id}.png"),
            "question": norm_text(qa_result.get("question", item["question"])),
            "answer": norm_text(qa_result.get("answer", item["answer"])),
            "evidence_bbox": None,  # VisR-Bench VI không có bbox
            "question_type": qa_result.get("question_type", "text"),
            "split": "dev",
            "annotator_model": MODEL,
        }

        labeled.append(label)
        print(f"  Question (Qwen): {label['question'][:60]}...")
        print(f"  Answer (Qwen): {label['answer']}")
        print(f"  Type: {label['question_type']}")

        # Save checkpoint every BATCH_SIZE
        if (idx + 1) % BATCH_SIZE == 0:
            checkpoint_file = OUT_DIR / f"qa_vi_haiku_full_checkpoint_{idx+1:05d}.jsonl"
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                for lbl in labeled:
                    f.write(json.dumps(lbl, ensure_ascii=False) + "\n")
            print(f"\nCheckpoint saved: {checkpoint_file.name}")

    # Save final batch
    out_file = OUT_DIR / "qa_vi_haiku_full_batch_100.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for lbl in labeled:
            f.write(json.dumps(lbl, ensure_ascii=False) + "\n")

    print(f"\n✓ Labeled {len(labeled)} items")
    print(f"✓ Saved to {out_file.relative_to(ROOT)}")
    print(f"\n** STOP HERE FOR MANUAL QA CHECK **")
    print(f"Review {len(labeled)} items, then continue full batch if OK.")


if __name__ == "__main__":
    main()
