"""Bước 1 của HAVC: chẩn đoán head "biết đọc chữ" bằng tác vụ OCR (Xie và cs., arXiv:2601.22483, mục 3).

Paper: với mỗi token đầu ra y_i mà mô hình đoán đúng, mỗi head h có phân bố attention A_h^(i) trên token đầu vào;
đỉnh j* = argmax_{j ∈ Ω'} A_h^(i)[j] (Ω' = token đầu vào trừ token đặc biệt). ProjScore = 1/‖m_i‖₁ nếu đỉnh
rơi vào vùng chữ m_i của token, ngược lại 0. Score_h = trung bình trên các token đúng; chuẩn hóa min-max trên
toàn ma trận, giữ head > 0,5. Hàm ở đây chỉ ghi lại đỉnh và vùng chữ của từng token; điểm tính ở analyze_heads.py.

Chỗ paper không nói rõ, tự chọn (phải nêu trong báo cáo):
  - Teacher forcing: đưa văn bản đúng làm câu trả lời, một lần forward cho cả ảnh; "đọc đúng" = argmax logits tại
    vị trí trước token bằng đúng token đó. Attention A_h^(i) lấy từ hàng truy vấn ở vị trí đó.
  - Ω' = các vị trí của prompt (ảnh + câu lệnh), bỏ token đặc biệt; không gồm các token trả lời trước đó.
  - Xét mọi layer × mọi head.
  - m_i = các ô lưới token ảnh (sau gộp 2x2 của Qwen2.5-VL, mỗi ô 28x28 px) chồng lên khung mực của TỪ chứa
    token. Lưới của Qwen2.5-VL là H/28 x W/28 (không vuông như N_p x N_p của LLaVA trong paper).
  - Cùng một câu lệnh tiếng Anh cho cả hai bộ ảnh, để chỉ ngôn ngữ của chữ trong ảnh thay đổi.
"""

import re
import time
import unicodedata

import numpy as np

PROMPT = "Read all the text in the image exactly as written."
VI_MARK = re.compile(r"[̀-̣̉]|đ", re.IGNORECASE)


def has_mark(word: str) -> bool:
    return bool(VI_MARK.search(unicodedata.normalize("NFD", word)))


def build_inputs(processor, image, text):
    """Prompt (ảnh + câu lệnh) theo chat template, nối thêm token của văn bản đúng làm câu trả lời."""
    import torch
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(text=[prompt], images=[image], return_tensors="pt")
    enc = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    n_prompt = inp["input_ids"].shape[1]
    ans = torch.tensor([enc["input_ids"]], dtype=inp["input_ids"].dtype)
    for k in list(inp.keys()):  # mọi tensor theo chiều chuỗi (input_ids, attention_mask, mm_token_type_ids...)
        v = inp[k]
        if hasattr(v, "shape") and v.dim() == 2 and v.shape == (1, n_prompt):
            fill = ans if k == "input_ids" else (torch.ones_like(ans) if k == "attention_mask" else torch.zeros_like(ans))
            inp[k] = torch.cat([v, fill.to(v.dtype)], 1)
    return inp, n_prompt, enc["input_ids"], enc["offset_mapping"]


def token_words(text, words, offsets):
    """Chỉ số từ (theo thứ tự vẽ) của từng token; -1 nếu token chỉ là khoảng trắng/xuống dòng."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    assert [text[a:b] for a, b in spans] == [w[0] for w in words], "văn bản và danh sách từ lệch nhau"
    owner = np.full(len(text) + 1, -1)
    for k, (a, b) in enumerate(spans):
        owner[a:b] = k
    out = []
    for s, e in offsets:
        k = next((owner[c] for c in range(s, e) if owner[c] >= 0), -1)
        out.append(int(k))
    return out


def word_rect(box, size, grid):
    """Khung pixel (x0, y0, x1, y1) trên ảnh gốc -> ô lưới (r0, r1, c0, c1) mà khung chồng lên (gồm hai đầu)."""
    (W, H), (gh, gw) = size, grid
    x0, y0, x1, y1 = box
    c0, c1 = int(x0 * gw / W), int(np.ceil(x1 * gw / W)) - 1
    r0, r1 = int(y0 * gh / H), int(np.ceil(y1 * gh / H)) - 1
    clip = lambda v, hi: max(0, min(hi - 1, v))
    return clip(r0, gh), max(clip(r0, gh), clip(r1, gh)), clip(c0, gw), max(clip(c0, gw), clip(c1, gw))


def diagnose_image(model, processor, image, meta):
    """Một ảnh -> dict mảng theo token: đỉnh attention (layer, head) và vùng chữ của token.

    peaks[t, l, h] = chỉ số ô lưới (r * gw + c) của đỉnh, -1 nếu đỉnh rơi vào token chữ (không phải ảnh).
    """
    import torch
    inp, n_prompt, ids, offsets = build_inputs(processor, image, meta["text"])
    inp = inp.to(model.device)
    tok = processor.tokenizer
    image_id = model.config.image_token_id
    special = set(tok.all_special_ids) - {image_id}
    prompt_ids = inp["input_ids"][0, :n_prompt]
    valid = torch.tensor([i not in special for i in prompt_ids.tolist()], device=prompt_ids.device)
    vis_pos = (prompt_ids == image_id).nonzero().flatten()
    t, gh2, gw2 = inp["image_grid_thw"][0].tolist()
    gh, gw = gh2 // 2, gw2 // 2
    assert t == 1 and len(vis_pos) == gh * gw, (t, gh, gw, len(vis_pos))
    cell_of = torch.full((n_prompt,), -1, dtype=torch.long, device=prompt_ids.device)
    cell_of[vis_pos] = torch.arange(gh * gw, device=prompt_ids.device)

    L = inp["input_ids"].shape[1]
    q = torch.arange(n_prompt - 1, L - 1, device=prompt_ids.device)  # hàng truy vấn dự đoán từng token trả lời
    t0 = time.time()
    with torch.no_grad():
        out = model(**inp, output_attentions=True)
    # 2 GPU (device_map="auto"): logits nằm ở card cuối, input_ids ở card đầu -> so trên CPU
    pred = out.logits[0, q.to(out.logits.device)].argmax(-1).cpu()
    correct = (pred == inp["input_ids"][0, n_prompt:].cpu()).numpy()
    peaks = []
    for a in out.attentions:  # mỗi layer: (1, heads, L, L)
        a = a[0][:, q.to(a.device)][:, :, :n_prompt].float()
        a = a.masked_fill(~valid.to(a.device), -1.0)
        peaks.append(cell_of.to(a.device)[a.argmax(-1)].cpu())  # (heads, n_tok)
    peaks = torch.stack(peaks).permute(2, 0, 1).numpy().astype(np.int16)  # (n_tok, layers, heads)
    del out
    dt = time.time() - t0

    owner = token_words(meta["text"], meta["words"], offsets)
    rects = np.array([word_rect(meta["words"][k][1:5], meta["size"], (gh, gw)) if k >= 0 else (-1, -1, -1, -1)
                      for k in owner], dtype=np.int16)
    mark = np.array([k >= 0 and has_mark(meta["words"][k][0]) for k in owner])
    keep = np.array(owner) >= 0
    return {"peaks": peaks[keep], "rect": rects[keep], "correct": correct[keep], "mark": mark[keep],
            "word": np.array(owner)[keep].astype(np.int32), "token": np.array(ids)[keep].astype(np.int32),
            "grid": (gh, gw), "n_tokens": len(ids), "seconds": dt}


def collect(model, processor, meta_rows, image_root, out_path, log_every=50):
    """Chạy diagnose_image cho từng ảnh, ghép theo token và lưu .npz. Trả về thống kê thời gian/VRAM."""
    import torch
    from PIL import Image
    parts, secs = [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i, m in enumerate(meta_rows):
        with Image.open(f"{image_root}/{m['image']}") as im:
            r = diagnose_image(model, processor, im.convert("RGB"), m)
        n = len(r["word"])
        r["image"] = np.full(n, i, dtype=np.int32)
        r["grid"] = np.tile(np.array(r["grid"], dtype=np.int16), (n, 1))
        parts.append(r); secs.append(r["seconds"])
        if (i + 1) % log_every == 0:
            acc = np.concatenate([p["correct"] for p in parts]).mean()
            print(f"  {i + 1}/{len(meta_rows)} ảnh, {np.mean(secs):.2f} s/ảnh, đọc đúng {acc:.1%} token")
    keys = ["peaks", "rect", "correct", "mark", "word", "token", "image", "grid"]
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    data["image_ids"] = np.array([m["id"] for m in meta_rows])
    np.savez_compressed(out_path, **data)
    stats = {"n_images": len(meta_rows), "n_tokens_kept": int(len(data["word"])),
             "seconds_per_image": round(float(np.mean(secs)), 3), "wall_seconds": round(time.time() - t0, 1),
             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
             "read_acc_token": round(float(data["correct"].mean()), 4)}
    return stats
