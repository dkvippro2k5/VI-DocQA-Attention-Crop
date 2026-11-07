"""Phân tích kết quả chẩn đoán head (head_scores.py) cho hai bộ ảnh tiếng Anh và tiếng Việt.

Điểm theo paper HAVC: ProjScore = 1/|m_i| nếu đỉnh rơi vào vùng chữ, Score_h = trung bình trên token đọc đúng,
chuẩn hóa min-max trên cả ma trận layer x head, giữ head > 0,5 (THRESH), xếp hạng lấy K = 8.
Kèm tỉ lệ trúng (đỉnh rơi vào vùng chữ, không chia |m|) vì dễ diễn giải hơn.

Câu hỏi của pilot (scope mục 9, rủi ro "VLM 1-3B định vị chữ tiếng Việt kém ở mọi head"):
  1. Mô hình đọc đúng bao nhiêu token mỗi ngôn ngữ; có head nào định vị tốt chữ tiếng Việt không.
  2. Tập head chọn bằng tiếng Anh và bằng tiếng Việt trùng nhau đến đâu (Jaccard, Spearman).
  3. Trên ảnh tiếng Việt (nửa ảnh giữ lại, không dùng để chọn head): head chọn bằng tiếng Việt định vị tốt hơn
     head chọn bằng tiếng Anh không, tách riêng từ có dấu / không dấu. Chọn head trên ảnh chẵn, đánh giá trên ảnh lẻ.

Chạy từ gốc repo:
    python -X utf8 -m src.diagnosis.analyze_heads results/C4_havc_en/diagnosis_en.npz results/C5_havc_vi/diagnosis_vi.npz
"""

import argparse
import json
import sys

import numpy as np

K = 8
THRESH = 0.5


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def hits(d):
    """(T, layers, heads) bool: đỉnh rơi vào vùng chữ của token."""
    p, r, gw = d["peaks"].astype(np.int32), d["rect"].astype(np.int32), d["grid"][:, 1].astype(np.int32)
    row, col = p // gw[:, None, None], p % gw[:, None, None]
    ok = (p >= 0)
    ok &= (row >= r[:, 0, None, None]) & (row <= r[:, 1, None, None])
    ok &= (col >= r[:, 2, None, None]) & (col <= r[:, 3, None, None])
    return ok


def head_scores(d, sel):
    """Score_h theo paper (trung bình ProjScore trên token đọc đúng trong sel) và tỉ lệ trúng."""
    h = hits(d)[sel]
    area = ((d["rect"][:, 1] - d["rect"][:, 0] + 1) * (d["rect"][:, 3] - d["rect"][:, 2] + 1)).astype(float)[sel]
    proj = h / area[:, None, None]
    return proj.mean(0), h.mean(0)


def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def top(score, k=K):
    flat = np.argsort(score.ravel())[::-1][:k]
    return [tuple(int(v) for v in np.unravel_index(i, score.shape)) for i in flat]


def spearman(a, b):
    ra, rb = a.ravel().argsort().argsort(), b.ravel().argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def summarize(en, vi):
    out = {}
    sets = {}
    for name, d in (("en", en), ("vi", vi)):
        c = d["correct"].astype(bool)
        score, rate = head_scores(d, c)
        norm = minmax(score)
        sets[name] = {"score": score, "rate": rate, "top": top(score), "above": top(norm, int((norm > THRESH).sum()))}
        out[name] = {
            "n_images": int(len(np.unique(d["image"]))), "n_tokens": int(len(c)),
            "read_acc_token": round(float(c.mean()), 4),
            "read_acc_marked_words": round(float(c[d["mark"].astype(bool)].mean()), 4) if d["mark"].any() else None,
            "read_acc_unmarked_words": round(float(c[~d["mark"].astype(bool)].mean()), 4),
            "n_heads_above_thresh": int((norm > THRESH).sum()),
            "top_k": [{"layer": l, "head": h, "score_norm": round(float(norm[l, h]), 3),
                       "hit_rate": round(float(rate[l, h]), 3)} for l, h in sets[name]["top"]],
            "best_hit_rate": round(float(rate.max()), 3),
            "mean_hit_rate_all_heads": round(float(rate.mean()), 4),
            # đỉnh rơi ngẫu nhiên đều trên các ô ảnh thì trúng với xác suất |m| / số ô
            "chance_hit_rate": round(float((((d["rect"][:, 1] - d["rect"][:, 0] + 1) * (d["rect"][:, 3] - d["rect"][:, 2] + 1))
                                            / (d["grid"][:, 0].astype(float) * d["grid"][:, 1]))[c].mean()), 4),
            "layers_of_above_thresh": sorted({l for l, _ in sets[name]["above"]}),
        }
    out["overlap"] = {
        "jaccard_top_k": round(jaccard(sets["en"]["top"], sets["vi"]["top"]), 3),
        "jaccard_above_thresh": round(jaccard(sets["en"]["above"], sets["vi"]["above"]), 3),
        "spearman_score": round(spearman(sets["en"]["score"], sets["vi"]["score"]), 3),
        "common_top_k": sorted(set(sets["en"]["top"]) & set(sets["vi"]["top"])),
    }

    # chọn head trên ảnh chẵn, đánh giá định vị trên token tiếng Việt đọc đúng của ảnh lẻ
    def half(d, parity):
        return d["correct"].astype(bool) & (d["image"] % 2 == parity)
    en_top = top(head_scores(en, half(en, 0))[0])
    vi_top = top(head_scores(vi, half(vi, 0))[0])
    rng = np.random.default_rng(0)
    L, H = vi["peaks"].shape[1:]
    rand = [[(int(i // H), int(i % H)) for i in rng.choice(L * H, K, replace=False)] for _ in range(200)]
    hv = hits(vi)
    held = half(vi, 1)
    groups = {"tất cả": held, "từ có dấu": held & vi["mark"].astype(bool), "từ không dấu": held & ~vi["mark"].astype(bool)}

    def rate_of(heads, sel):  # tỉ lệ trúng trung bình của K head
        return float(np.mean([hv[sel, l, h].mean() for l, h in heads])) if sel.any() else None
    out["vi_heldout"] = {g: {"n_tokens": int(sel.sum()),
                             "heads_from_vi": round(rate_of(vi_top, sel), 4) if sel.any() else None,
                             "heads_from_en": round(rate_of(en_top, sel), 4) if sel.any() else None,
                             "random_8_mean": round(float(np.mean([rate_of(r, sel) for r in rand])), 4) if sel.any() else None}
                         for g, sel in groups.items()}
    out["vi_heldout"]["note"] = "chọn head trên ảnh chẵn, đánh giá trên ảnh lẻ; con số là tỉ lệ đỉnh attention rơi vào từ"
    out["vi_heldout_bootstrap"] = bootstrap_vi_minus_en(hv, vi_top, en_top, vi["image"], groups)
    return out


def bootstrap_vi_minus_en(hv, vi_top, en_top, image, groups, n_boot=5000, seed=0):
    """Paired bootstrap theo ảnh cho hiệu (head chọn bằng tiếng Việt - head chọn bằng tiếng Anh) trên ảnh giữ lại.

    Lấy mẫu lại theo ảnh vì các token trong một ảnh không độc lập. Thêm hiệu-của-hiệu (có dấu - không dấu):
    nếu dương thì lợi thế của head tiếng Việt lớn hơn ở từ có dấu. p hai phía; Holm trên 4 kiểm định.
    """
    diff = (np.mean([hv[:, l, h] for l, h in vi_top], 0) - np.mean([hv[:, l, h] for l, h in en_top], 0))
    imgs, inv = np.unique(image[groups["tất cả"]], return_inverse=True)
    idx = np.full(image.max() + 1, -1)
    idx[imgs] = np.arange(len(imgs))
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, len(imgs), len(imgs)), minlength=len(imgs)) for _ in range(n_boot)])

    def per_image(sel):  # tổng hiệu và số token của từng ảnh giữ lại
        i = idx[image[sel]]
        return np.bincount(i, diff[sel], len(imgs)), np.bincount(i, minlength=len(imgs)).astype(float)

    est, boots = {}, {}
    for g in ("tất cả", "từ có dấu", "từ không dấu"):
        s, c = per_image(groups[g])
        est[g], boots[g] = s.sum() / c.sum(), (w @ s) / (w @ c)
    est["có dấu - không dấu"] = est["từ có dấu"] - est["từ không dấu"]
    boots["có dấu - không dấu"] = boots["từ có dấu"] - boots["từ không dấu"]

    res = {g: {"diff_vi_minus_en": round(float(est[g]), 4),
               "ci95": [round(float(v), 4) for v in np.percentile(boots[g], [2.5, 97.5])],
               "p_boot": float(min(1.0, 2 * min((boots[g] <= 0).mean(), (boots[g] >= 0).mean())))}
           for g in est}
    # Holm
    order = sorted(res, key=lambda g: res[g]["p_boot"])
    running = 0.0
    for r, g in enumerate(order):
        running = max(running, min(1.0, (len(order) - r) * res[g]["p_boot"]))
        res[g]["p_holm"] = round(running, 4)
    res["note"] = f"paired bootstrap theo ảnh, {n_boot} lần, {len(imgs)} ảnh lẻ; p_boot = 0 nghĩa là < 1/{n_boot}"
    return res


def expert_heads(d, source):
    """Expert head cho HAVC lúc inference: điểm chuẩn hóa > THRESH, tính trên mọi token đọc đúng, xếp giảm dần."""
    norm = minmax(head_scores(d, d["correct"].astype(bool))[0])
    heads = top(norm, int((norm > THRESH).sum()))
    return {"source": source, "thresh": THRESH, "n_heads": len(heads),
            "heads": [[l, h] for l, h in heads], "score_norm": [round(float(norm[l, h]), 4) for l, h in heads]}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("en")
    ap.add_argument("vi")
    ap.add_argument("--out", help="ghi tóm tắt ra file JSON")
    ap.add_argument("--export-heads", action="store_true",
                    help="ghi expert_heads_config.json cạnh mỗi file .npz (đầu vào của C4/C5)")
    a = ap.parse_args()
    en, vi = load(a.en), load(a.vi)
    res = summarize(en, vi)
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    print(txt)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(txt)
    if a.export_heads:
        from pathlib import Path
        for path, d in ((a.en, en), (a.vi, vi)):
            p = Path(path).with_name("expert_heads_config.json")
            p.write_text(json.dumps(expert_heads(d, Path(path).name), indent=1), encoding="utf-8")
            print("Đã ghi", p)


if __name__ == "__main__":
    main()
