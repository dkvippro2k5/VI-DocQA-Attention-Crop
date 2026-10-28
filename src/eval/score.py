"""Chấm điểm đáp án VLM: EM, F1 theo âm tiết, và so sánh ghép cặp giữa hai cột dự đoán.

Chuẩn hóa: NFC, chữ thường, bỏ dấu câu, gộp khoảng trắng. KHÔNG bỏ dấu thanh.

Chạy từ gốc repo:
    python -m src.eval.score results/C3_vicrop/dev_raw.jsonl pred_full pred_crop
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


EDGE_PUNCT = ".,;:!?\"'()[]“”‘’…"


def normalize(s: str) -> str:
    """Chỉ bỏ dấu câu ở đầu/cuối từ; giữ ký hiệu như '+', '++' (đáp án chuẩn có thể chỉ là ký hiệu)."""
    s = unicodedata.normalize("NFC", s).lower().replace("_", " ")
    toks = [t.strip(EDGE_PUNCT) for t in s.split()]
    return " ".join(t for t in toks if t)


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def contains_gold(pred: str, gold: str) -> int:
    """Nới lỏng: đáp án chuẩn nằm trọn trong dự đoán (VLM hay trả dài hơn)."""
    g = normalize(gold)
    return int(bool(g) and g in normalize(pred))


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    path, col_a, col_b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    per_q = []
    for r in rows:
        rec = {"qid": r["qid"], "gold": r["gold"]}
        for col in (col_a, col_b):
            rec[col] = {
                "em": exact_match(r[col], r["gold"]),
                "contains": contains_gold(r[col], r["gold"]),
                "f1": round(f1(r[col], r["gold"]), 4),
            }
        per_q.append(rec)

    n = len(per_q)
    summary = {"n": n}
    for col in (col_a, col_b):
        summary[col] = {m: round(sum(q[col][m] for q in per_q) / n, 4) for m in ("em", "contains", "f1")}

    for m in ("contains", "f1"):
        win = sum(q[col_b][m] > q[col_a][m] for q in per_q)
        loss = sum(q[col_b][m] < q[col_a][m] for q in per_q)
        summary[f"paired_{m}"] = {f"{col_b}_better": win, f"{col_a}_better": loss, "tie": n - win - loss}

    out = path.with_name(path.stem + "_scored.json")
    out.write_text(json.dumps({"summary": summary, "per_question": per_q}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nChi tiết từng câu: {out}")


if __name__ == "__main__":
    main()
