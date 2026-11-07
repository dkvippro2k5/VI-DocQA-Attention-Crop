"""HAVC lúc inference (nhánh C4, C5) cho Qwen2.5-VL: lọc expert head theo entropy không gian và độ nhạy gradient,
gộp K head thành bản đồ hướng dẫn, rồi lấy hộp crop như ViCrop.

Theo Xie và cs., arXiv:2601.22483, mục 2.2 (đã đọc toàn văn phần phương pháp):
  - Entropy không gian (Eq. 5): bản đồ attention của head chuẩn hóa về [0,1], nhị phân bằng Otsu, tách C_h thành
    phần liên thông; E_h = min(λ_c (C_h - 1) + λ_d · d̄_h / d_max, 1), d̄_h = khoảng cách trung bình giữa các tâm
    thành phần, d_max = đường chéo bản đồ. Chỉ giữ head có E_h < 0,3.
  - Độ nhạy gradient (Eq. 6-7): S_h^sens = ∂ log p(y*) / ∂ a_h, G_h = <a_h, max(0, S_h^sens)>.
  - Gộp (Eq. 8): S_h = α N(1 - E_h) + (1 - α) N(G_h), N = min-max trên các head; lấy top-K theo S_h.
  - Trọng số (Eq. 9-10): w_h = softmax(S_h / τ) trên K head; M_final = Σ w_h A_h. Hộp lấy từ M_final theo ViCrop.
  Siêu tham số theo paper: K = 8, α = 0,4, τ = 0,1, ngưỡng entropy 0,3 (ngưỡng 0,5 của bước chọn expert head nằm
  ở src/diagnosis/analyze_heads.py).

Chỗ paper không nói rõ, tự chọn (phải nêu trong báo cáo):
  - λ_c, λ_d: paper ghi "theo [8]" (Kang và cs., CVPR 2025) nhưng [8] dùng công thức entropy Shannon khác, không
    có λ. Tự đặt LAMBDA_C, LAMBDA_D bên dưới; đã lưu bản đồ từng head để xem lại trên dev.
  - Liên thông 8 hướng (theo [8]); paper HAVC không nói.
  - a_h = attention của token đầu vào cuối cùng (vị trí sinh token đầu tiên) lên các token ảnh, như ViCrop;
    y* = token đầu tiên mô hình đoán (argmax). Paper chỉ nói "truy vấn giải mã hiện tại" và "token dự đoán".
  - Bản đồ thô, không chia cho attention của prompt chung như rel_att của ViCrop (paper không nhắc).
  - Entropy và gradient chỉ tính trên expert head của bước chẩn đoán.
  - Còn ít hơn K head qua ngưỡng entropy thì dùng hết số còn lại; không còn head nào thì lấy K head entropy thấp
    nhất và đánh dấu fallback.
  - Đạo hàm nhân GRAD_SCALE trước backward để tránh underflow ở fp16, rồi chia lại; N(G_h) không đổi.
"""

import numpy as np

K = 8
ALPHA = 0.4
TAU = 0.1
ENTROPY_THRESH = 0.3
LAMBDA_C = 0.1  # tự chọn, paper không cho giá trị
LAMBDA_D = 0.5  # tự chọn, paper không cho giá trị
GRAD_SCALE = 1024.0


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Ngưỡng Otsu cho giá trị trong [0, 1]: cực đại phương sai giữa hai lớp."""
    hist, edges = np.histogram(x.ravel(), bins=bins, range=(0.0, 1.0))
    p = hist.astype(float) / max(1, hist.sum())
    mid = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * mid)
    w1 = 1.0 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1
    return float(edges[int(np.argmax(between)) + 1])


def spatial_entropy(att_map: np.ndarray, lam_c: float = LAMBDA_C, lam_d: float = LAMBDA_D):
    """Trả về (E_h, C_h, d̄_h / d_max) của một bản đồ attention (H_grid, W_grid)."""
    from scipy import ndimage
    m = att_map.astype(float)
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    labels, c = ndimage.label(m > otsu_threshold(m), structure=np.ones((3, 3)))
    if c < 2:
        return 0.0, int(c), 0.0
    cent = np.array(ndimage.center_of_mass(np.ones_like(m), labels, range(1, c + 1)))
    dist = np.sqrt(((cent[:, None] - cent[None]) ** 2).sum(-1))
    d_rel = float(dist[np.triu_indices(c, 1)].mean() / np.hypot(*m.shape))
    return float(min(lam_c * (c - 1) + lam_d * d_rel, 1.0)), int(c), d_rel


def _minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def head_maps_and_grads(model, processor, image, prompt: str, heads):
    """Một lần forward + backward. Trả về (maps {(l, h): (H_grid, W_grid)}, grads {(l, h): G_h}, y*).

    Cần attn_implementation='eager' và tham số model đã requires_grad_(False) (chỉ cần gradient theo attention).
    Gradient chạy được nhờ hook đặt requires_grad cho embedding đầu vào; phần vision encoder không giữ đồ thị.
    """
    import torch
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    vis = (inp["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
    t, gh2, gw2 = inp["image_grid_thw"][0].tolist()
    grid = (gh2 // 2, gw2 // 2)
    assert t == 1 and len(vis) == grid[0] * grid[1], (t, grid, len(vis))

    layers = sorted({l for l, _ in heads})
    hook = model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    try:
        with torch.enable_grad():
            out = model(**inp, output_attentions=True)
            logp = out.logits[0, -1].float().log_softmax(-1)
            y = int(logp.argmax())
            atts = [out.attentions[l] for l in layers]
            grads = torch.autograd.grad(logp[y] * GRAD_SCALE, atts)
    finally:
        hook.remove()

    maps, g_h = {}, {}
    for l, a, g in zip(layers, atts, grads):
        v = vis.to(a.device)
        a_q = a[0, :, -1, v].detach().float()  # (heads, n_vis)
        g_q = g[0, :, -1, v].float() / GRAD_SCALE
        sens = (a_q * g_q.clamp(min=0)).sum(-1).cpu().numpy()
        a_q = a_q.cpu().numpy().reshape(-1, *grid)
        for (ll, h) in heads:
            if ll == l:
                maps[(l, h)], g_h[(l, h)] = a_q[h], float(sens[h])
    del out, atts, grads
    return maps, g_h, y


def guidance_map(maps, grads, heads, k=K, alpha=ALPHA, tau=TAU, e_thresh=ENTROPY_THRESH,
                 lam_c=LAMBDA_C, lam_d=LAMBDA_D):
    """Lọc entropy, gộp với gradient, lấy top-K, trả về (M_final, info). heads: expert head theo thứ tự chẩn đoán."""
    ent = {h: spatial_entropy(maps[h], lam_c, lam_d) for h in heads}
    kept = [h for h in heads if ent[h][0] < e_thresh]
    fallback = not kept
    if fallback:
        kept = sorted(heads, key=lambda h: ent[h][0])[:k]
    s = alpha * _minmax(np.array([1 - ent[h][0] for h in kept])) + \
        (1 - alpha) * _minmax(np.array([grads[h] for h in kept]))
    order = np.argsort(-s, kind="stable")[:k]
    top = [kept[i] for i in order]
    w = np.exp((s[order] - s[order].max()) / tau)
    w /= w.sum()
    m_final = sum(wi * maps[h] for wi, h in zip(w, top))
    info = {"n_experts": len(heads), "n_pass_entropy": 0 if fallback else len(kept), "fallback": fallback,
            "top_heads": [[int(l), int(h)] for l, h in top], "weights": [round(float(x), 4) for x in w],
            "entropy_top": [round(ent[h][0], 3) for h in top],
            "median_components": float(np.median([ent[h][1] for h in heads]))}
    return m_final, info
