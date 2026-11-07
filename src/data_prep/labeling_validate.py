"""Kiểm tra, gộp đầu ra của worker gán đáp án, soạn tin nhắc việc, đo độ lệch giữa worker.

Lỗi (loại dòng): JSON hỏng, thiếu trường, qid lạ/trùng/thuộc batch khác, kiểu sai, answerable và answer mâu thuẫn.
Cảnh báo (giữ dòng, gắn cờ): đáp án quá dài, không tìm thấy trong text trang (trừ evidence_type=figure).
"Thiếu" chỉ tính trong các batch đã có file đầu ra. Có vấn đề thì in sẵn tin NHẮC LẠI để dán cho worker.
Batch hiệu chuẩn `a00_anchor` (chế độ nhiều worker): mọi worker làm cùng các câu; mỗi câu giữ đáp án đồng thuận
nhất (medoid theo F1), in độ khớp giữa các worker và worker lệch nhất.

Chạy từ gốc repo (sau mỗi batch), liệt kê thư mục outputs của một hoặc nhiều worker:
    python -X utf8 -m src.data_prep.labeling_validate <outputs_1> [<outputs_2> ...] --annotator "<model> (Antigravity)" --out <file.jsonl>
        [--compare data/processed/visr_bench_vi/qa_vi_visr_answers.jsonl] [--out đường_dẫn.jsonl]
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from src.data_prep.label_visr_answers import collect_questions, excluded_docs
from src.eval.score import f1, normalize

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "visr_bench_vi"
ANCHOR_ID = "a00_anchor"
REQUIRED = ("qid", "answer", "answerable", "evidence_type", "worker")
EVIDENCE = {"text", "table", "figure"}
MAX_WORDS = 25
PROMPT_VERSION = "v3"


def page_norm(text: str) -> str:
    return normalize(re.sub(r"[*_#|>`\-]+", " ", text))


def no_accent(s: str) -> str:
    return unicodedata.normalize("NFD", s).replace("đ", "d").encode("ascii", "ignore").decode()


def in_page(ans: str, page_text: str) -> bool:
    """page_md của tài liệu không có PDF thường bị mất dấu ('BO GIAO THONG VAN TAI'): khớp đúng hoặc khớp khi bỏ
    dấu đều tính là có trên trang. Việc kiểm dấu thanh do người duyệt làm, cờ này chỉ để bắt đáp án bịa."""
    a, p = normalize(ans), page_norm(page_text)
    return a in p or no_accent(a) in no_accent(p)


def check_row(row: dict, truth: dict) -> tuple[list, list]:
    errs, warns = [], []
    if "evidence_type" not in row:  # trường phụ (không dùng khi duyệt hay chấm): thiếu thì giữ dòng, gắn cờ
        row["evidence_type"] = None
        warns.append("thiếu_evidence_type")
    miss = [k for k in REQUIRED if k not in row]
    if miss:
        return [f"thiếu trường {miss}"], warns
    if row["qid"] not in truth:
        return ["qid không có trong manifest"], warns
    if not isinstance(row["answerable"], bool):
        errs.append("answerable không phải bool")
    if not isinstance(row["answer"], str):
        return errs + ["answer không phải chuỗi"], warns
    if "thiếu_evidence_type" in warns:
        pass
    elif row["answerable"] is False and row["evidence_type"] in (None, "", "none"):
        row["evidence_type"] = None  # TASK.md không quy định evidence_type cho câu không có đáp án; worker hay ghi none/null
    elif row["evidence_type"] not in EVIDENCE:
        errs.append(f"evidence_type lạ: {row['evidence_type']!r}")
    if not str(row["worker"]).strip():
        errs.append("worker rỗng")
    if "agent" in row and not str(row["agent"]).strip():
        errs.append("agent rỗng")
    ans = unicodedata.normalize("NFC", row["answer"]).strip()
    if isinstance(row["answerable"], bool):
        if row["answerable"] and not ans:
            errs.append("answerable=true nhưng answer rỗng")
        if not row["answerable"] and ans:
            errs.append("answerable=false nhưng answer không rỗng")
    if ans:
        if len(ans.split()) > MAX_WORDS and ";" not in ans:  # v3: đáp án nhiều ý ngăn bằng "; " được phép dài
            warns.append("dài")
        if row["evidence_type"] != "figure" and not in_page(ans, truth[row["qid"]]["page_text"]):
            warns.append("không_có_trong_text_trang")
    return errs, warns


def batch_index(outputs_dirs) -> dict:
    """{tên batch: [qid theo thứ tự manifest]} từ thư mục manifests nằm cạnh mỗi thư mục outputs."""
    idx = {}
    for d in outputs_dirs:
        for f in sorted((Path(d).parent / "manifests").glob("*.jsonl")):
            qs = [json.loads(l)["qid"] for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            idx.setdefault(f.stem, qs)
    return idx


def reminder(errors, flagged, missing_by_batch, qid_batch, who="") -> str:
    """Tin nhắc cho MỘT worker. Cờ "không khớp text trang" không đưa vào (text trích thường mất dấu hoặc sai OCR,
    nhắc theo nó gây sửa nhầm); cờ đó chỉ dùng để xếp ưu tiên lúc duyệt tay."""
    lines = [f"NHẮC LẠI{' cho ' + who if who else ''} (dán nguyên văn cho worker này):",
             "Hãy đọc lại TASK.md từ đầu. Kết quả bạn vừa ghi còn các vấn đề sau. Chỉ sửa đúng các dòng nêu dưới đây,",
             "ghi lại trong cùng file outputs/<batch>.jsonl, không đổi các dòng khác."]
    if errors:
        lines.append("\n1) Lỗi định dạng (bắt buộc sửa):")
        lines += [f"  - {where}: {msg}" for where, msg in errors[:20]]
    if missing_by_batch:
        lines.append("\n2) Còn thiếu dòng cho các qid:")
        for b, qs in missing_by_batch.items():
            lines.append(f"  - {b}: {', '.join(qs[:10])}" + (f" (+{len(qs) - 10} qid nữa)" if len(qs) > 10 else ""))
    if flagged.get("dài"):
        lines.append(f"\n3) Đáp án quá {MAX_WORDS} từ mà không ngăn ý bằng \"; \". Câu hỏi nhiều ý thì ngăn các ý bằng"
                     " \"; \"; một ý thì rút gọn thành cụm ngắn nhất chép nguyên văn:")
        lines += [f"  - [{qid_batch.get(q, '?')}] {q}" for q in flagged["dài"][:20]]
    return "\n".join(lines)


def medoid(answers: dict) -> str:
    """Agent có đáp án khớp nhiều nhất với các agent khác (tổng F1); hòa thì agent đứng trước."""
    best, best_s = None, -1.0
    for a in sorted(answers):
        s = sum(f1(answers[a], answers[b]) for b in answers if b != a)
        if s > best_s:
            best, best_s = a, s
    return best


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs_dirs", nargs="+", help="Một hay nhiều thư mục outputs (mỗi worker một thư mục)")
    ap.add_argument("--annotator", required=True, help="Công cụ/model gán, ghi vào annotator_model")
    ap.add_argument("--compare")
    # Bắt buộc ghi rõ đích: trước đây mặc định là file nhãn vòng 1, quên --out khi chạy vòng 2 đã ghi đè mất nó.
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-version", default=PROMPT_VERSION, help="Chỉ đổi khi dựng lại nhãn của vòng cũ (vòng 1: v2)")
    args = ap.parse_args()

    truth = {i["qid"]: i for i in collect_questions()}
    skip_docs = excluded_docs()
    is_excluded = lambda qid: str(qid).split("_")[0] in skip_docs
    idx = {b: [q for q in qs if not is_excluded(q)] for b, qs in batch_index(args.outputs_dirs).items()}
    idx = {b: qs for b, qs in idx.items() if qs}
    n_skipped = 0
    qid_batch = {q: b for b, qs in idx.items() for q in qs}
    files = sorted(f for d in args.outputs_dirs for f in Path(d).glob("*.jsonl"))
    if not files:
        sys.exit(f"Không có file .jsonl trong {args.outputs_dirs}")

    def make_row(row, t, warns):
        return {
            "qid": row["qid"], "document_id": t["document_id"], "page_id": t["page_id"],
            "image": t["image"], "question": t["question"],
            "answer": unicodedata.normalize("NFC", row["answer"]).strip(),
            "answerable": row["answerable"], "evidence_type": row["evidence_type"],
            "question_type": None, "evidence_bbox": None, "split": "",
            "annotator_model": args.annotator, "worker": str(row["worker"]).strip(),
            "agent": str(row.get("agent", "")).strip() or None,
            "prompt_version": args.prompt_version, "flags": warns,
        }

    kept, errors, seen = [], [], set()
    anchor = defaultdict(dict)  # qid -> {agent: row đã dựng}
    for f in files:
        is_anchor = f.stem == ANCHOR_ID
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{f.parent.parent.name}/{f.name}:{ln}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((where, f"JSON hỏng: {e}"))
                continue
            if isinstance(row, dict) and is_excluded(row.get("qid", "")):
                n_skipped += 1  # tài liệu nằm trong data/qa/excluded_documents.json
                continue
            errs, warns = check_row(row, truth)
            if not errs and f.stem in idx and row["qid"] not in idx[f.stem]:
                errs = [f"qid thuộc batch khác ({qid_batch.get(row['qid'], '?')}), không thuộc {f.stem}"]
            if not errs and is_anchor and not str(row.get("agent", "")).strip():
                errs = ["batch hiệu chuẩn thiếu trường agent"]
            if not errs and is_anchor:
                ag = str(row["agent"]).strip()
                if ag in anchor[row["qid"]]:
                    errs = [f"qid trùng trong cùng agent {ag}"]
                else:
                    anchor[row["qid"]][ag] = make_row(row, truth[row["qid"]], warns)
                    continue
            elif not errs and row["qid"] in seen:
                errs = ["qid trùng"]
            if errs:
                errors.append((where, "; ".join(errs)))
                continue
            seen.add(row["qid"])
            kept.append(make_row(row, truth[row["qid"]], warns))

    # Batch hiệu chuẩn: gộp thành một đáp án đồng thuận cho mỗi câu và đo độ khớp giữa các agent
    calib_rows = [r for d in anchor.values() for r in d.values()]
    pair_f1 = defaultdict(list)
    lows = []
    for qid, by_ag in anchor.items():
        ans_ok = {a: r["answer"] for a, r in by_ag.items() if r["answerable"]}
        chosen = by_ag[medoid(ans_ok)] if ans_ok else sorted(by_ag.items())[0][1]
        chosen = dict(chosen, flags=chosen["flags"] + ["hiệu_chuẩn"])
        kept.append(chosen)
        seen.add(qid)
        agents = sorted(ans_ok)
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                s = f1(ans_ok[a], ans_ok[b])
                pair_f1[a].append(s)
                pair_f1[b].append(s)
                lows.append((s, qid, a, ans_ok[a], b, ans_ok[b]))
        if len({r["answerable"] for r in by_ag.values()}) > 1:
            lows.append((0.0, qid, "-", "worker bất đồng về answerable", "-", str({a: r["answerable"] for a, r in by_ag.items()})))

    started = {f.stem for f in files if f.stem in idx and f.stem != ANCHOR_ID}
    missing_by_batch = {b: [q for q in idx[b] if q not in seen] for b in sorted(started)}
    missing_by_batch = {b: qs for b, qs in missing_by_batch.items() if qs}
    not_started = [b for b in idx if b not in started and b != ANCHOR_ID]
    flagged = defaultdict(list)
    for r in kept + calib_rows:
        for fl in r["flags"]:
            if fl != "hiệu_chuẩn":
                flagged[fl].append(r["qid"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")

    n_missing = sum(len(v) for v in missing_by_batch.values())
    workers = Counter(r["worker"] for r in kept + calib_rows)
    print(f"Hợp lệ: {len(kept)}/{len(truth)} | lỗi (bị loại): {len(errors)} | thiếu trong batch đã làm: {n_missing} "
          f"| batch chưa làm: {len(not_started)}")
    if n_skipped:
        print(f"Bỏ qua {n_skipped} dòng thuộc tài liệu bị loại (data/qa/excluded_documents.json)")
    print(f"answerable=false: {sum(not r['answerable'] for r in kept)} | cờ: {({k: len(set(v)) for k, v in flagged.items()}) or 'không'}")
    print(f"Tên model tự khai (chỉ tham khảo, không đáng tin; tên chuẩn là --annotator \"{args.annotator}\"): {dict(workers)}")
    print(f"Đã ghi {out}")

    if anchor:
        agents_done = sorted({a for d in anchor.values() for a in d})
        allf = [s for v in pair_f1.values() for s in v]
        print(f"\nHiệu chuẩn: {len(anchor)} câu, agent đã nộp: {agents_done}")
        if allf:
            print(f"  Độ khớp F1 giữa các cặp agent: trung bình {sum(allf) / len(allf):.3f} (mỗi cặp tính một lần, đã tính hai chiều)")
            for a in sorted(pair_f1):
                print(f"  {a}: khớp trung bình với agent khác {sum(pair_f1[a]) / len(pair_f1[a]):.3f}")
            print("  5 cặp lệch nhất (đọc tay để xem lệch kiểu gì):")
            for s, qid, a, aa, b, ab in sorted(lows, key=lambda x: x[0])[:5]:
                print(f"    F1={s:.2f} {qid[-14:]} | {a}: {aa[:50]} | {b}: {ab[:50]}")

    # Tin nhắc tách theo worker (thư mục chứa outputs); batch hiệu chuẩn không nhắc để giữ đo độ lệch độc lập.
    owner = {}
    for d in args.outputs_dirs:
        for f in (Path(d).parent / "manifests").glob("*.jsonl"):
            owner.setdefault(f.stem, set()).add(Path(d).parent.name)
    for who in sorted({Path(d).parent.name for d in args.outputs_dirs}):
        mine = lambda b: b != ANCHOR_ID and who in owner.get(b, ())
        w_err = [(wh, m) for wh, m in errors if wh.split("/")[0] == who and mine(wh.split("/")[1].split(".")[0])]
        w_miss = {b: qs for b, qs in missing_by_batch.items() if mine(b)}
        # Cờ "dài" không đưa vào tin nhắc: các đáp án này thường là một câu định nghĩa chép nguyên văn, rút gọn sẽ mất
        # nội dung. Cờ vẫn được lưu để người duyệt quyết định (tick L hoặc bỏ câu).
        w_flag = {}
        if w_err or w_miss:
            print("\n" + reminder(w_err, w_flag, w_miss, qid_batch, who))

    if args.compare:
        other = {}
        for line in Path(args.compare).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                other[d["qid"]] = d["answer"]
        pairs = [(r, other[r["qid"]]) for r in kept if r["qid"] in other and r["answerable"]]
        if not pairs:
            print("\nSo sánh: không có qid chung")
            return
        scores = [(f1(r["answer"], o), r, o) for r, o in pairs]
        mean = sum(s for s, _, _ in scores) / len(scores)
        exact = sum(normalize(r["answer"]) == normalize(o) for _, r, o in scores)
        print(f"\nSo sánh với {Path(args.compare).name}: {len(scores)} câu chung | F1 trung bình {mean:.3f} | khớp hoàn toàn {exact}")
        print("15 cặp lệch nhất (để đọc tay):")
        for s, r, o in sorted(scores, key=lambda x: x[0])[:15]:
            print(f"  F1={s:.2f} | {r['question'][:55]}\n      mới : {r['answer'][:70]}\n      cũ  : {o[:70]}")


if __name__ == "__main__":
    main()
