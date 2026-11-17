"""Kiểm định ghép cặp giữa các nhánh trên cùng bộ câu hỏi (scope mục 9).

- Chỉ số nhị phân (em, contains, em_nodiac): McNemar chính xác (nhị thức trên các cặp bất đồng), kèm paired bootstrap.
- Chỉ số liên tục (f1, item_cover, f1_nodiac): paired bootstrap trên hiệu từng câu, khoảng tin cậy 95% theo phân vị.
- So sánh chính (--primary, khai báo trước: C5 vs C4) báo cáo p chưa hiệu chỉnh.
  Các so sánh phụ (--secondary) hiệu chỉnh Holm, theo từng chỉ số.
- Bootstrap mặc định lấy mẫu lại theo câu, như scope ghi. Câu cùng tài liệu không độc lập, nên có thêm
  --cluster document_id (lấy mẫu lại theo tài liệu) làm phân tích độ nhạy; báo cáo cả hai nếu kết luận khác nhau.
  McNemar luôn giả định các câu độc lập.
- Các nhánh phải có đúng cùng tập qid và cùng đáp án chuẩn, nếu không sẽ dừng với lỗi.

Mỗi nhánh khai báo dạng TÊN=đường_dẫn.jsonl:cột_dự_đoán; so sánh dạng B:A (hiệu = B − A).
Chạy từ gốc repo, ví dụ:
    python -m src.eval.stats \
        --branch C4=results/C4_havc_en/dev_raw.jsonl:pred --branch C5=results/C5_havc_vi/dev_raw.jsonl:pred \
        --branch B0=results/B0_base/dev_raw.jsonl:pred \
        --primary C5:C4 --secondary C5:B0 --secondary C4:B0 \
        --subset answer_form=short --out results/stats/dev_short.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from src.eval.score import answer_form, contains_gold, exact_match, f1, item_cover, normalize, strip_diacritics

METRICS = {
    "em": lambda p, g: exact_match(p, g),
    "contains": lambda p, g: contains_gold(p, g),
    "f1": lambda p, g: f1(p, g),
    "item_cover": lambda p, g: item_cover(p, g),
    "em_nodiac": lambda p, g: exact_match(strip_diacritics(p), strip_diacritics(g)),
    "f1_nodiac": lambda p, g: f1(strip_diacritics(p), strip_diacritics(g)),
}
BINARY = {"em", "contains", "em_nodiac"}


def load_branch(spec: str) -> tuple[str, dict]:
    name, rest = spec.split("=", 1)
    path, col = rest.rsplit(":", 1)
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["qid"] in rows:
            sys.exit(f"[{name}] qid lặp: {r['qid']}")
        rows[r["qid"]] = {
            "pred": r.get(col) or "",
            "gold": r["gold"],
            # Bộ công khai có nhiều đáp án chấp nhận (TextVQA, DocVQA, ViTextVQA); chấm lấy điểm cao nhất
            "answers": r.get("answers") or [r["gold"]],
            "document_id": r.get("document_id") or r["qid"].split("_")[0],
            "answer_form": r.get("answer_form") or answer_form(r["gold"], bool(r.get("long_answer"))),
            "question_type": r.get("question_type"),
        }
    return name, {"path": path, "column": col, "rows": rows}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: mảng 0/1 cùng câu. Trả số câu chỉ A đúng, chỉ B đúng và p hai phía (nhị thức chính xác)."""
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_A_correct": only_a, "only_B_correct": only_b, "p": p}


def paired_bootstrap(d: np.ndarray, clusters: np.ndarray | None, n_boot: int, rng: np.random.Generator) -> dict:
    """d: hiệu từng câu (B − A). Trả trung bình, CI 95% và p hai phía xấp xỉ từ phân phối bootstrap."""
    if clusters is None:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        means = d[idx].mean(axis=1)
    else:
        uniq, inv = np.unique(clusters, return_inverse=True)
        sums = np.bincount(inv, weights=d, minlength=len(uniq))
        counts = np.bincount(inv, minlength=len(uniq))
        pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # (k + 1) / (B + 1): không bao giờ ra p = 0; p nhỏ nhất là 2 / (B + 1)
    p = min(1.0, 2 * (min((means <= 0).sum(), (means >= 0).sum()) + 1) / (n_boot + 1))
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p": float(p)}


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", action="append", required=True, help="TÊN=file.jsonl:cột")
    ap.add_argument("--primary", help="B:A, so sánh chính khai báo trước")
    ap.add_argument("--secondary", action="append", default=[], help="B:A, so sánh phụ (hiệu chỉnh Holm)")
    ap.add_argument("--metrics", default="em,f1,em_nodiac,f1_nodiac")
    ap.add_argument("--subset", action="append", default=[], help="trường=giá_trị, ví dụ answer_form=short")
    ap.add_argument("--cluster", choices=["document_id"], help="bootstrap theo tài liệu (phân tích độ nhạy)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    branches = dict(load_branch(s) for s in args.branch)
    names = list(branches)
    qids = set(branches[names[0]]["rows"])
    for n in names[1:]:
        if set(branches[n]["rows"]) != qids:
            diff = qids ^ set(branches[n]["rows"])
            sys.exit(f"Nhánh {n} khác tập câu hỏi với {names[0]} ({len(diff)} qid lệch, ví dụ {sorted(diff)[:3]}).")
        for q in qids:
            if normalize(branches[n]["rows"][q]["gold"]) != normalize(branches[names[0]]["rows"][q]["gold"]):
                sys.exit(f"Đáp án chuẩn của {q} khác nhau giữa {names[0]} và {n}.")

    ref = branches[names[0]]["rows"]
    for cond in args.subset:
        key, val = cond.split("=", 1)
        qids = {q for q in qids if str(ref[q].get(key)) == val}
    qids = sorted(qids)
    if not qids:
        sys.exit("Không còn câu nào sau khi lọc --subset.")

    metrics = args.metrics.split(",")
    def score(m: str, pred: str, answers: list[str]) -> float:
        return max(METRICS[m](pred, g) for g in answers)

    scores = {n: {m: np.array([score(m, branches[n]["rows"][q]["pred"], ref[q]["answers"]) for q in qids], dtype=float)
                  for m in metrics} for n in names}
    clusters = np.array([ref[q]["document_id"] for q in qids]) if args.cluster else None
    if clusters is not None and len(set(clusters)) < 10:
        print(f"CẢNH BÁO: chỉ có {len(set(clusters))} tài liệu; bootstrap theo tài liệu không đáng tin "
              "(khoảng tin cậy hẹp giả tạo).\n")
    rng = np.random.default_rng(args.seed)

    def compare(spec: str) -> dict:
        b, a = spec.split(":")
        res = {"B": b, "A": a, "metrics": {}}
        for m in metrics:
            r = {"mean_A": float(scores[a][m].mean()), "mean_B": float(scores[b][m].mean())}
            r["bootstrap"] = paired_bootstrap(scores[b][m] - scores[a][m], clusters, args.n_boot, rng)
            if m in BINARY:
                r["mcnemar"] = mcnemar_exact(scores[a][m], scores[b][m])
            res["metrics"][m] = r
        return res

    out = {
        "config": {"branches": {n: {"path": branches[n]["path"], "column": branches[n]["column"]} for n in names},
                   "subset": args.subset, "n_questions": len(qids), "n_documents": len({ref[q]["document_id"] for q in qids}),
                   "metrics": metrics, "n_boot": args.n_boot, "seed": args.seed, "cluster": args.cluster},
        "means": {n: {m: float(scores[n][m].mean()) for m in metrics} for n in names},
        "primary": compare(args.primary) if args.primary else None,
        "secondary": [compare(s) for s in args.secondary],
    }
    # Holm cho họ so sánh phụ, tách theo chỉ số; p dùng McNemar với chỉ số nhị phân, bootstrap với chỉ số liên tục
    for m in metrics:
        key = "mcnemar" if m in BINARY else "bootstrap"
        ps = [c["metrics"][m][key]["p"] for c in out["secondary"]]
        for c, p_adj in zip(out["secondary"], holm(ps)):
            c["metrics"][m]["p_holm"] = p_adj

    print(f"{len(qids)} câu, {out['config']['n_documents']} tài liệu; lọc: {args.subset or 'không'}; "
          f"bootstrap theo {'tài liệu' if args.cluster else 'câu'}\n")
    print("Trung bình:", "  ".join(f"{n}: " + ", ".join(f"{m}={out['means'][n][m]:.3f}" for m in metrics) for n in names))
    for label, comps in (("CHÍNH", [out["primary"]] if out["primary"] else []), ("PHỤ", out["secondary"])):
        for c in comps:
            print(f"\n[{label}] {c['B']} − {c['A']}")
            for m, r in c["metrics"].items():
                bs = r["bootstrap"]
                line = f"  {m:10s} {bs['diff']:+.3f}  CI95 [{bs['ci95'][0]:+.3f}, {bs['ci95'][1]:+.3f}]  p_boot={bs['p']:.4f}"
                if "mcnemar" in r:
                    mc = r["mcnemar"]
                    line += f"  McNemar {mc['only_B_correct']}/{mc['only_A_correct']} p={mc['p']:.4f}"
                if "p_holm" in r:
                    line += f"  p_Holm={r['p_holm']:.4f}"
                print(line)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nGhi {args.out}")


if __name__ == "__main__":
    main()
