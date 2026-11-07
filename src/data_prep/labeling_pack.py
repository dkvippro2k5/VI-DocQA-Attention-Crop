"""Tạo gói công việc gán đáp án cho worker bên ngoài (ví dụ agent trong Antigravity), làm theo batch.

Hai chế độ:
- 1 worker (mặc định): label_tasks/, các batch làm tuần tự, không có batch hiệu chuẩn.
- N worker (--workers N): label_tasks_multi/worker_1..N. Mỗi worker một thư mục riêng, chỉ thấy batch của mình
  cộng một batch hiệu chuẩn `a00_anchor` chung cho mọi worker (câu giống nhau). So đáp án các worker trên batch
  này để đo độ lệch giữa worker (labeling_validate). Câu hiệu chuẩn không nằm trong batch thường.
Gói KHÔNG chứa đáp án cũ của model khác, để nhãn mới độc lập và so sánh được.

Vòng gán nhãn bổ sung (--round r2): thư mục label_tasks_r2/worker_1..N, chỉ gồm câu CHƯA có trong
qa_vi_visr_labels_external.jsonl (nhãn vòng trước), để không xóa hay làm lại đầu ra vòng trước. --anchors-max giới
hạn số câu hiệu chuẩn (chọn đều trong danh sách theo tài liệu), vì mỗi worker phải làm hết batch hiệu chuẩn.
page_text lấy từ lớp text PDF khi đọc được (xem label_visr_answers.pdf_page_texts), không thì page_md.

Chạy từ gốc repo:
    python -X utf8 -m src.data_prep.labeling_pack [--workers N]
    python -X utf8 -m src.data_prep.labeling_pack --workers 4 --round r2 --anchors-max 12
    python -X utf8 -m src.data_prep.labeling_pack --refresh-docs   # chỉ cập nhật hướng dẫn, giữ nguyên outputs/
Ra: data/processed/visr_bench_vi/label_tasks[_multi]/ (đã gitignore, tạo lại được).
"""

import json
import math
import shutil
import sys
from pathlib import Path

from src.data_prep.label_visr_answers import IN_DIR, collect_questions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "visr_bench_vi"
BATCH_MAX = 20
# v2: thêm quy tắc chép phần nội dung, không chép câu dẫn
# v3 (giữa vòng r2): bỏ trần 25 từ cho câu hỏi nhiều ý; đáp án dài không còn là lý do ghi answerable=false
PROMPT_VERSION = "v3"
ANCHOR_ID = "a00_anchor"

RULES = """- Chép nguyên văn từ trang, đúng dấu thanh và dấu mũ tiếng Việt; không dịch, không diễn đạt lại, không sửa lỗi của tài liệu.
- Ngắn nhất mà vẫn đủ ý; chỉ dùng thông tin trên trang, không suy đoán, không tìm trên mạng. Đáp án một ý: tối đa 25 từ.
- Câu hỏi nhiều ý ("những gì", "các ... nào", "trường hợp nào", liệt kê): chép ĐỦ mọi ý trang nêu, mỗi ý ngắn gọn,
  các ý ngăn bằng "; ", kể cả khi tổng dài hơn 25 từ. Đáp án dài KHÔNG phải lý do để ghi answerable=false.
- Câu hỏi về việc cần làm, lý do, hậu quả hay nội dung một mục: chép chính phần nội dung đó, không chép câu dẫn
  hay tiêu đề (kiểu "hãy đọc các hướng dẫn sau", "sau đây là", "như sau").
- Không có đáp án trên trang hoặc không chắc: answerable=false và answer="" (đừng đoán). Chỉ ghi false sau khi
  đã nhìn kỹ cả trang, kể cả bảng, chú thích, chữ nhỏ.
- Nhìn ảnh từng câu; text trang chỉ để tham khảo, khác ảnh thì tin ảnh; không viết script tự sinh đáp án.
- Đúng một dòng JSON cho mỗi qid, theo thứ tự manifest, UTF-8, không thêm chữ ngoài JSON.
- Trường `worker` (tên model) luôn cùng một giá trị cho mọi dòng và mọi batch."""


def task_md(agent):
    ex_agent = f', "agent": "{agent}"' if agent else ""
    agent_field = (f"\n- `agent`: luôn là `\"{agent}\"` (mã của bạn, không đổi).\n" if agent else "")
    who = f"Bạn là worker `{agent}`. " if agent else ""
    calib = ("\nBatch `a00_anchor` là batch hiệu chuẩn: các worker khác cũng làm đúng các câu này, độc lập với bạn. "
             "Làm nó như mọi batch khác, không làm khác đi.\n") if agent else ""
    return f"""# Nhiệm vụ gán đáp án (prompt {PROMPT_VERSION})

{who}Bạn là người gán đáp án cho bộ dữ liệu hỏi đáp tài liệu tiếng Việt. Mỗi câu hỏi đi kèm ảnh một trang
tài liệu. Hãy đọc ảnh và trả lời bằng đáp án **ngắn nhất, chép nguyên văn từ trang**.

Công việc chia thành nhiều batch nhỏ, làm lần lượt. Mỗi batch bạn được giao một file trong `manifests/`.
**Đầu mỗi batch hãy đọc lại toàn bộ file này**, đừng dựa vào trí nhớ của các batch trước.
{calib}
## Cách làm

1. Mỗi dòng trong `manifests/<batch>.jsonl` là một câu hỏi: `{{"qid", "question", "image", "page_text"}}`.
   `image` và `page_text` là đường dẫn tương đối.
2. Với từng câu hỏi, **mở và nhìn ảnh** trong `image`. `page_text` là text trích tự động, chỉ để tham khảo
   và có thể sai chính tả hoặc sai dấu; khi khác ảnh thì tin ảnh.
3. Ghi kết quả vào `outputs/<batch>.jsonl`, đúng **một dòng JSON cho mỗi qid** trong batch, theo thứ tự manifest.

## Quy tắc đáp án

{RULES}
- Không sửa manifest. Không đọc đầu ra của batch khác. Không có đáp án mẫu nào khác trong gói.

## Ví dụ về câu dẫn (minh họa, không thuộc dữ liệu cần gán)

Câu hỏi: "Cần làm gì trước khi vệ sinh máy?". Trang ghi: "Trước khi vệ sinh, hãy làm theo các bước sau: 1. Rút phích
cắm khỏi ổ điện. 2. Chờ máy nguội." Đáp án đúng là phần nội dung ("Rút phích cắm khỏi ổ điện"), không phải câu dẫn
("làm theo các bước sau").

## Định dạng mỗi dòng đầu ra

```json
{{"qid": "...", "answer": "1.800 hộ dân", "answerable": true, "evidence_type": "text", "worker": "<tên model của bạn>"{ex_agent}}}
```

- `qid`: giữ nguyên từ manifest.
- `answer`: chuỗi; rỗng nếu `answerable` là false.
- `answerable`: true hoặc false.
- `evidence_type`: `"text"` (đoạn văn), `"table"` (ô trong bảng) hoặc `"figure"` (hình, sơ đồ, ảnh).
- `worker`: tên và phiên bản model bạn đang chạy; ghi **giống hệt nhau ở mọi dòng và mọi batch**.{agent_field}
## Tự kiểm tra trước khi ghi xong

- Số dòng đầu ra bằng số dòng manifest, qid không thiếu, không trùng.
- Đáp án một ý không quá 25 từ, không phải câu giải thích dài khi chỉ cần một cụm; đáp án nhiều ý đủ các ý, ngăn bằng "; ".
- Mỗi câu answerable=false: đã tìm lại trên ảnh và chắc chắn trang không có đáp án.
- Mỗi đáp án đối chiếu lại với ảnh: đúng chữ, đúng dấu.
- `answerable=false` thì `answer` rỗng, và ngược lại.

## Khi xong batch

Báo: số dòng đã ghi, số câu `answerable=false`, và đáp án dài nhất (bao nhiêu từ). Người phụ trách sẽ chạy kiểm tra
tự động và có thể gửi lại danh sách qid cần sửa; khi đó chỉ sửa các dòng được nêu, trong cùng file đầu ra.
"""


def chunk(rows, label_start):
    rows = sorted(rows, key=lambda r: (r["page_id"], r["qid"]))
    k = math.ceil(len(rows) / BATCH_MAX)
    base, extra = divmod(len(rows), k)
    out, pos = [], 0
    for j in range(k):
        size = base + (1 if j < extra else 0)
        out.append(rows[pos:pos + size])
        pos += size
    return out


def make_batches(items):
    by_doc = {}
    for it in items:
        by_doc.setdefault(it["document_id"], []).append(it)
    batches, n = [], 0
    for doc_id, rows in sorted(by_doc.items()):
        for part in chunk(rows, n):
            n += 1
            batches.append((f"b{n:02d}_{doc_id[:8]}", part))
    return batches


def pick_anchors(items, max_n=None):
    """Chọn đều mỗi tài liệu vài câu (khoảng sqrt(n)/2,5, tối thiểu 1), xác định, không ngẫu nhiên.
    max_n: nếu nhiều hơn thì lấy đều max_n câu trong danh sách đã chọn (sắp theo tài liệu)."""
    by_doc = {}
    for it in items:
        by_doc.setdefault(it["document_id"], []).append(it)
    picked = []
    for doc_id, rows in sorted(by_doc.items()):
        rows = sorted(rows, key=lambda r: (r["page_id"], r["qid"]))
        q = min(len(rows), max(1, round(math.sqrt(len(rows)) / 2.5)))
        picked += [rows[int((j + 0.5) * len(rows) / q)] for j in range(q)]
    if max_n and len(picked) > max_n:
        picked = [picked[int((j + 0.5) * len(picked) / max_n)] for j in range(max_n)]
    return picked


def prompts_md(batches, agent, with_anchor):
    n = len(batches) + (1 if with_anchor else 0)
    who = f"Worker `{agent}`. " if agent else ""
    out = ["# Prompt dán cho worker\n",
           f"{who}Dùng **cùng một model** cho mọi batch (và mọi worker). Làm tuần tự; sau mỗi batch chạy script kiểm tra.\n",
           "## 0. Khởi động (gửi một lần, trước batch đầu tiên)\n", "```",
           (f"Bạn là worker {agent}. " if agent else "") + "Đọc TASK.md. Trước khi bắt đầu, hãy diễn đạt lại bằng lời của bạn 5 quy tắc",
           "đáp án quan trọng nhất và định dạng một dòng đầu ra, để tôi xác nhận bạn hiểu đúng. Chưa làm batch nào cả.",
           "```\n"]
    order = ([(ANCHOR_ID, None)] if with_anchor else []) + [(bid, rows) for bid, rows in batches]
    for i, (bid, rows) in enumerate(order, 1):
        size = len(rows) if rows is not None else "?"
        out += [f"## Batch {i}/{n}: `{bid}`\n", "```",
                f"Batch {i}/{n}. Hãy đọc lại TASK.md từ đầu (đừng dựa vào trí nhớ các batch trước).",
                "Nhắc nhanh các quy tắc:", RULES]
        if agent:
            out.append(f"- Trường `agent` luôn là \"{agent}\".")
        out += [f"Việc cần làm: đọc manifests/{bid}.jsonl, nhìn từng ảnh, ghi outputs/{bid}.jsonl.",
                "Chỉ dùng TASK.md, manifest này, images/ và pages/.",
                "Khi xong báo: số dòng đã ghi, số câu answerable=false, đáp án dài nhất (bao nhiêu từ).", "```\n"]
    out += ["## Khi worker lệch việc\n",
            "Chạy `labeling_validate` sau mỗi batch. Nếu có lỗi hoặc cờ cảnh báo, script in sẵn tin **NHẮC LẠI** liệt kê "
            "đúng qid cần sửa; dán nguyên văn cho worker. Nếu lỗi lặp lại sau khi nhắc, mở phiên mới (cùng model, cùng "
            "prompt khởi động) rồi làm tiếp từ batch đó.\n"]
    return "\n".join(out)


def write_pack(dest, batches, agent, anchors):
    """batches: [(bid, rows)]; anchors: rows của batch hiệu chuẩn hoặc None."""
    if dest.exists():
        shutil.rmtree(dest)
    for sub in ("manifests", "images", "pages", "outputs"):
        (dest / sub).mkdir(parents=True)
    named = ([(ANCHOR_ID, anchors)] if anchors else []) + list(batches)
    info = {}
    for bid, rows in named:
        man = []
        for r in rows:
            img_rel = f"images/{r['document_id']}_{r['page_id']}.png"
            txt_rel = f"pages/{r['document_id']}_p{r['page_id']}.md"
            if not (dest / img_rel).exists():
                shutil.copy(IN_DIR / r["image"], dest / img_rel)
            (dest / txt_rel).write_text(r["page_text"], encoding="utf-8")
            man.append({"qid": r["qid"], "question": r["question"], "image": img_rel, "page_text": txt_rel})
        (dest / "manifests" / f"{bid}.jsonl").write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man), encoding="utf-8")
        info[bid] = len(man)
    (dest / "TASK.md").write_text(task_md(agent), encoding="utf-8")
    (dest / "PROMPTS.md").write_text(prompts_md(batches, agent, bool(anchors)), encoding="utf-8")
    (dest / "outputs" / ".gitkeep").write_text("", encoding="utf-8")
    (dest / "batches.json").write_text(json.dumps({"prompt_version": PROMPT_VERSION, "agent": agent, "batches": info},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def refresh_docs() -> None:
    """Ghi lại TASK.md, PROMPTS.md, batches.json trong mọi gói đã có; KHÔNG đụng manifests/, images/, outputs/."""
    for dest in sorted(list(OUT.glob("label_tasks_multi/worker_*")) + list(OUT.glob("label_tasks_r*/worker_*"))
                       + [OUT / "label_tasks"]):
        meta = dest / "batches.json"
        if not meta.exists():
            continue
        old = json.loads(meta.read_text(encoding="utf-8"))
        agent = old.get("agent")
        names = [b for b in old["batches"] if b != ANCHOR_ID]
        (dest / "TASK.md").write_text(task_md(agent), encoding="utf-8")
        (dest / "PROMPTS.md").write_text(prompts_md([(b, []) for b in names], agent, ANCHOR_ID in old["batches"]), encoding="utf-8")
        old["prompt_version"] = PROMPT_VERSION
        meta.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"đã làm mới TASK.md, PROMPTS.md ({PROMPT_VERSION}): {dest.relative_to(ROOT)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if "--refresh-docs" in sys.argv:
        refresh_docs()
        return
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    workers = int(arg("--workers", 1))
    rnd = arg("--round")
    anchors_max = int(arg("--anchors-max", 0)) or None
    items = collect_questions()
    labeled = OUT / "qa_vi_visr_labels_external.jsonl"
    if rnd:
        done = {json.loads(l)["qid"] for l in labeled.read_text(encoding="utf-8").splitlines() if l.strip()}
        items = [i for i in items if i["qid"] not in done]
        print(f"Vòng {rnd}: bỏ {len(done)} câu đã gán ở vòng trước, còn {len(items)} câu")

    if workers <= 1:
        info = write_pack(OUT / "label_tasks", make_batches(items), None, None)
        print(f"1 worker: {sum(info.values())} câu, {len(info)} batch -> {(OUT / 'label_tasks').relative_to(ROOT)}")
        return

    anchors = pick_anchors(items, anchors_max)
    anchor_ids = {a["qid"] for a in anchors}
    batches = make_batches([i for i in items if i["qid"] not in anchor_ids])
    loads, owned = [0] * workers, [[] for _ in range(workers)]
    for bid, rows in sorted(batches, key=lambda b: -len(b[1])):
        k = loads.index(min(loads))
        owned[k].append((bid, rows))
        loads[k] += len(rows)
    root = OUT / (f"label_tasks_{rnd}" if rnd else "label_tasks_multi")
    if rnd and root.exists() and any(root.glob("worker_*/outputs/*.jsonl")):
        sys.exit(f"{root.relative_to(ROOT)} đã có đầu ra của worker; không ghi đè. Xóa tay nếu thật sự muốn tạo lại.")
    if root.exists():
        shutil.rmtree(root)
    print(f"{workers} worker | {len(items)} câu = {len(items) - len(anchors)} thường + {len(anchors)} câu hiệu chuẩn (mỗi worker làm cả {len(anchors)})")
    for k in range(workers):
        bs = sorted(owned[k], key=lambda b: b[0])
        write_pack(root / f"worker_{k + 1}", bs, f"w{k + 1}", anchors)
        print(f"  worker_{k + 1} (w{k + 1}): {loads[k]} câu + {len(anchors)} hiệu chuẩn, batch {[b for b, _ in bs]}")
    print(f"Thư mục: {root.relative_to(ROOT)}")
    if rnd:
        dirs = " ".join(f"{root.relative_to(ROOT).as_posix()}/worker_{k + 1}/outputs" for k in range(workers))
        print("Kiểm tra sau mỗi batch (ghi ra file riêng, KHÔNG ghi đè nhãn vòng trước):")
        print(f"  python -X utf8 -m src.data_prep.labeling_validate {dirs} --annotator \"<model> (Antigravity)\" "
              f"--out data/processed/visr_bench_vi/qa_vi_visr_labels_external_{rnd}.jsonl")


if __name__ == "__main__":
    main()
