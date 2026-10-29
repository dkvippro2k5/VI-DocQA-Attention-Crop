"""ViCrop `rel_att` cho Qwen2.5-VL (tự cài lại, nhánh C3).

Bám cơ chế của saccharomycetes/mllms_know (qwen2_5_methods.py, utils.py), nhưng
được cài lại từ phần mô tả code gốc, chưa đối chiếu từng dòng. Cần diff với
bản gốc trước khi báo cáo; mọi chỗ khác gốc phải ghi vào báo cáo.

Cơ chế:
  1. Forward 2 lần cùng ảnh: prompt câu hỏi và prompt chung.
  2. Lấy attention của token cuối tới các token ảnh ở layer ATT_LAYER, trung bình theo head.
  3. rel_att = att_câu_hỏi / att_prompt_chung (khử thiên lệch vị trí).
  4. Reshape về lưới (h/2, w/2), tìm cửa sổ có tổng attention cao nhất ở nhiều tỉ lệ.
"""

import numpy as np

ATT_LAYER = 22  # mặc định của repo gốc
RATIOS = (1, 1.2, 1.4, 1.6, 1.8, 2)
GENERAL_PROMPT = "Write a general description of the image."  # [cần đối chiếu với repo gốc]


def bbox_from_att_map(att_map: np.ndarray, image_size: tuple, bbox_size: int):
    """Chọn hộp vuông quanh vùng attention cao nhất.

    att_map: (H_grid, W_grid). image_size: (W, H) pixel. bbox_size: cạnh hộp cơ sở (pixel).
    Trả về (x1, y1, x2, y2) pixel, nằm trong ảnh.
    """
    W, H = image_size
    gh, gw = att_map.shape
    bw, bh = W / gw, H / gh  # kích thước 1 ô lưới theo pixel

    best = None  # (độ nhọn, tâm_x, tâm_y, cạnh)
    for r in RATIOS:
        side = bbox_size * r
        nx, ny = max(1, round(side / bw)), max(1, round(side / bh))
        nx, ny = min(nx, gw), min(ny, gh)

        # tổng attention của mọi cửa sổ nx x ny, qua ảnh tích phân
        ii = np.pad(att_map, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = ii[ny:, nx:] - ii[:-ny, nx:] - ii[ny:, :-nx] + ii[:-ny, :-nx]
        y, x = np.unravel_index(np.argmax(sums), sums.shape)
        peak = sums[y, x]

        # độ nhọn: đỉnh so với trung bình các cửa sổ lân cận, chuẩn hóa theo diện tích
        neigh = [sums[j, i] for j in (y - 1, y, y + 1) for i in (x - 1, x, x + 1)
                 if (j, i) != (y, x) and 0 <= j < sums.shape[0] and 0 <= i < sums.shape[1]]
        sharp = (peak - (np.mean(neigh) if neigh else 0.0)) / (nx * ny)
        if best is None or sharp > best[0]:
            best = (sharp, (x + nx / 2) * bw, (y + ny / 2) * bh, side)

    _, cx, cy, side = best
    side = min(side, W, H)
    cx = min(max(cx, side / 2), W - side / 2)
    cy = min(max(cy, side / 2), H - side / 2)
    return (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))


def rel_att_map(model, processor, image, question_prompt: str, layer: int = ATT_LAYER):
    """Trả về rel_att (H_grid, W_grid) của một layer."""
    return rel_att_maps(model, processor, image, question_prompt, (layer,))[layer]


def rel_att_maps(model, processor, image, question_prompt: str, layers=(ATT_LAYER,)):
    """Trả về {layer: rel_att (H_grid, W_grid)}. Một lần forward cho mọi layer.

    Chạy trên GPU, cần attn_implementation='eager'.
    """
    import torch

    def run(prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        ids = inp["input_ids"][0]
        tok = processor.tokenizer
        pos = (ids == tok.convert_tokens_to_ids("<|vision_start|>")).nonzero()[0].item() + 1
        end = (ids == tok.convert_tokens_to_ids("<|vision_end|>")).nonzero()[0].item()
        grid = (inp["image_grid_thw"][0, 1:] // 2).tolist()
        return {l: out.attentions[l][0, :, -1, pos:end].mean(dim=0).float().cpu().numpy().reshape(grid)
                for l in layers}

    q, g = run(question_prompt), run(GENERAL_PROMPT)
    return {l: q[l] / (g[l] + 1e-8) for l in layers}
