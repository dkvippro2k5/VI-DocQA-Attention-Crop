"""Lấy mẫu dev/test cho ba bộ công khai của thiết kế 2 × 2 (scope mục 7.1): TextVQA val, DocVQA val, ViTextVQA test.

Quy tắc (chốt 15/11/2026, trước khi chạy nhánh nào trên các bộ này):
- Nhóm theo ảnh (TextVQA, ViTextVQA) hoặc theo tài liệu (DocVQA: ucsf_document_id), để dev và test không chung
  ảnh/tài liệu, giống quy tắc chia theo document_id của bộ tiếng Việt.
- Xáo các nhóm bằng seed cố định, lấy N_DEV + N_TEST nhóm đầu; mỗi nhóm lấy ngẫu nhiên MỘT câu. Một câu mỗi ảnh
  giúp các câu gần độc lập với nhau (McNemar và bootstrap theo câu giả định điều này).
- N_DEV nhóm đầu vào dev, N_TEST nhóm sau vào test.
- `answers` là các đáp án được chấp nhận; khi chấm lấy điểm cao nhất trên các đáp án. `answer` là đáp án xuất hiện
  nhiều nhất. TextVQA chỉ nhận đáp án có >= 2/10 người cùng trả lời (giữ toàn bộ ở `answers_raw`); DocVQA và
  ViTextVQA nhận mọi đáp án (ViTextVQA: gộp các annotation cùng ảnh, cùng câu hỏi).

Đầu ra: data/qa/public/<bộ>.jsonl (commit được) và ảnh của các câu được chọn ở data/raw/<bộ>/sampled/ (không commit).
Chạy từ gốc repo:
    python -m src.data_prep.sample_public --dataset textvqa
"""

import argparse
import functools
import json
import random
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SEED = 20261115
N_DEV, N_TEST = 100, 500
RAW = Path("data/raw")
OUT = Path("data/qa/public")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def majority(answers: list[str]) -> str:
    return Counter(answers).most_common(1)[0][0]


def load_parquet_meta(files: list[Path], cols: list[str]) -> list[dict]:
    rows = []
    for f in files:
        t = pq.read_table(f, columns=cols)
        for i, r in enumerate(t.to_pylist()):
            r["_file"], r["_row"] = str(f), i
            rows.append(r)
    return rows


@functools.lru_cache(maxsize=1)
def _image_column(file: str):
    return pq.read_table(file, columns=["image"]).column("image")


def parquet_image(file: str, row: int) -> bytes:
    """Đọc cả cột ảnh của một file một lần (cache 1 file); gọi theo thứ tự file để không đọc lại."""
    return _image_column(file)[row].as_py()["bytes"]


def textvqa() -> tuple[list[dict], callable]:
    files = sorted((RAW / "textvqa/data").glob("validation-*.parquet"))
    rows = load_parquet_meta(files, ["image_id", "question_id", "question", "answers"])
    items = [{"group": r["image_id"], "qid": f"textvqa_{r['question_id']}", "document_id": r["image_id"],
              "question": nfc(r["question"]), "answers": [nfc(a) for a in r["answers"]],
              "_src": (r["_file"], r["_row"])} for r in rows]
    return items, lambda it: parquet_image(*it["_src"])


def docvqa() -> tuple[list[dict], callable]:
    files = sorted((RAW / "docvqa/DocVQA").glob("validation-*.parquet"))
    rows = load_parquet_meta(files, ["questionId", "question", "question_types", "docId", "ucsf_document_id",
                                     "ucsf_document_page_no", "answers"])
    items = [{"group": r["ucsf_document_id"], "qid": f"docvqa_{r['questionId']}",
              "document_id": r["ucsf_document_id"], "page_id": f"{r['ucsf_document_id']}_p{r['ucsf_document_page_no']}",
              "question": nfc(r["question"]), "answers": [nfc(a) for a in r["answers"]],
              "source_question_types": r["question_types"], "_src": (r["_file"], r["_row"])} for r in rows]
    return items, lambda it: parquet_image(*it["_src"])


def vitextvqa() -> tuple[list[dict], callable]:
    d = RAW / "vitextvqa"
    # ViTextVQA_test.json để trống đáp án ("your answer"); đáp án nằm ở ViTextVQA_test_gt.json.
    gt = json.loads((d / "ViTextVQA_test_gt.json").read_text(encoding="utf-8"))
    test_imgs = {im["id"]: im["filename"] for im in gt["images"]}
    train = json.loads((d / "ViTextVQA_train.json").read_text(encoding="utf-8"))
    overlap = test_imgs.keys() & {im["id"] for im in train["images"]}
    if overlap:
        sys.exit(f"ViTextVQA: {len(overlap)} ảnh test cũng nằm trong train; dừng để kiểm tra.")
    # Một câu hỏi có nhiều đáp án hợp lệ được lưu thành nhiều annotation riêng (cùng ảnh, cùng câu hỏi);
    # gộp lại thành một câu với danh sách đáp án.
    merged = defaultdict(list)
    for a in gt["annotations"]:
        if a["image_id"] not in test_imgs:
            sys.exit(f"ViTextVQA: annotation {a['id']} trỏ tới ảnh {a['image_id']} không có trong tập test.")
        merged[(a["image_id"], nfc(a["question"]))].append(a)
    zf = zipfile.ZipFile(d / "ViTextVQA_images.zip")
    items = [{"group": img_id, "qid": f"vitextvqa_{min(a['id'] for a in anns)}", "document_id": str(img_id),
              "question": q, "answers": [nfc(x) for a in anns for x in a["answers"]],
              "_src": ("zip", f"st_images/{test_imgs[img_id]}")} for (img_id, q), anns in merged.items()]
    return items, lambda it: zf.read(it["_src"][1])


LOADERS = {"textvqa": textvqa, "docvqa": docvqa, "vitextvqa": vitextvqa}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(LOADERS), required=True)
    args = ap.parse_args()

    items, get_image = LOADERS[args.dataset]()
    groups = defaultdict(list)
    for it in items:
        if it["answers"] and it["question"]:
            groups[it["group"]].append(it)
    keys = sorted(groups)
    rng = random.Random(SEED)
    rng.shuffle(keys)
    if len(keys) < N_DEV + N_TEST:
        sys.exit(f"Chỉ có {len(keys)} nhóm, không đủ {N_DEV + N_TEST}.")

    img_dir = RAW / args.dataset / "sampled"
    img_dir.mkdir(parents=True, exist_ok=True)
    picked = [(k_i, rng.choice(sorted(groups[k], key=lambda x: x["qid"])))
              for k_i, k in enumerate(keys[:N_DEV + N_TEST])]
    # Ghi ảnh theo thứ tự nguồn để mỗi file parquet / zip chỉ phải đọc một lần
    paths = {}
    for _, it in sorted(picked, key=lambda p: str(p[1]["_src"])):
        existing = list(img_dir.glob(f"{it['qid']}.*"))
        if existing:
            paths[it["qid"]] = existing[0]
            continue
        data = get_image(it)
        ext = ".png" if data[:4] == b"\x89PNG" else ".jpg"
        paths[it["qid"]] = img_dir / f"{it['qid']}{ext}"
        paths[it["qid"]].write_bytes(data)
    out_rows = []
    for k_i, it in picked:
        rec = {k2: v for k2, v in it.items() if not k2.startswith("_") and k2 != "group"}
        cnt = Counter(it["answers"])
        if args.dataset == "textvqa":
            # 10 người trả lời mỗi câu; chỉ nhận đáp án có >= 2 người cùng trả lời (thước đo gốc cần 3 người cho
            # điểm tối đa) để một đáp án gõ nhầm của một người không được tính đúng. Không có thì lấy đa số.
            accepted = sorted(a for a, c in cnt.items() if c >= 2) or [majority(it["answers"])]
            rec["answers_raw"] = it["answers"]
        else:
            accepted = sorted(cnt)
        rec.update({"dataset": args.dataset, "image": paths[it["qid"]].as_posix(),
                    "answer": majority(it["answers"]), "answers": accepted,
                    "split": "dev" if k_i < N_DEV else "test"})
        out_rows.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.dataset}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    cfg = {"dataset": args.dataset, "seed": SEED, "n_dev": N_DEV, "n_test": N_TEST, "n_groups_total": len(keys),
           "n_questions_total": len(items), "group_by": "ucsf_document_id" if args.dataset == "docvqa" else "image"}
    (OUT / f"{args.dataset}_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(cfg, ensure_ascii=False), "->", out)


if __name__ == "__main__":
    main()
