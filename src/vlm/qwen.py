"""Nạp Qwen2.5-VL và sinh đáp án, dùng chung cho mọi nhánh (B0, B1, C3–C7).

Cấu hình mặc định khớp lượt C3 trên dev: fp16 (T4 không có bf16), eager attention (để mọi nhánh, kể cả nhánh
lấy attention, chạy cùng một cài đặt attention), max_pixels = 1024 * 28 * 28, giải mã tham lam.

Vision tower chạy fp32, phần LLM giữ fp16 (từ 17/11/2026). Lý do (notebooks/fp16_overflow_diag.ipynb,
results/diag_fp16/): ở fp16, output mlp.down_proj của vision block cuối (blocks.31) lên tới ~56 000 ngay cả ở câu
bình thường (ngưỡng fp16 là 65 504); ở 6/409 câu B0 dev nó vượt ngưỡng thành inf, logits thành NaN và model sinh
chuỗi "!!!!…" tới hết max_new_tokens. Đổi vision tower sang fp32 sửa cả 6 câu, 6 câu đối chứng giữ nguyên đáp án;
đổi 8 layer LLM cuối sang fp32 không sửa được câu nào. Chi phí đo trên T4: VRAM đỉnh thêm khoảng 1,4 GB, thêm khoảng
0,5–1 s/câu. Mọi nhánh phải dùng cùng cài đặt này (load_qwen, hoặc upcast_vision nếu notebook tự nạp model).
"""

import unicodedata

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MAX_PIXELS = 1024 * 28 * 28


def _cast_floats(x, dtype):
    if torch.is_tensor(x):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, dict):  # gồm cả ModelOutput
        for k in list(x.keys()):
            x[k] = _cast_floats(x[k], dtype)
        return x
    if isinstance(x, (tuple, list)):
        return type(x)(_cast_floats(v, dtype) for v in x)
    return x


def upcast_vision(model):
    """Đổi vision tower (gồm merger) sang fp32; đầu vào ép về fp32, đầu ra ép về dtype của LLM.
    Đầu ra merger nhỏ (đã qua LayerNorm và MLP chiếu), ép về fp16 không tràn."""
    visual = next(m for n, m in model.named_modules() if n.split(".")[-1] == "visual")
    out_dtype = next(p for n, p in model.named_parameters() if ".visual." not in f".{n}").dtype
    visual.float()
    visual.register_forward_pre_hook(
        lambda m, a, k: (_cast_floats(a, torch.float32), _cast_floats(k, torch.float32)), with_kwargs=True)
    visual.register_forward_hook(lambda m, a, o: _cast_floats(o, out_dtype))
    return model


def load_qwen(model_id: str = DEFAULT_MODEL, max_pixels: int = DEFAULT_MAX_PIXELS, attn: str = "eager",
              vision_fp32: bool = True):
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation=attn).eval()
    if vision_fp32:
        upcast_vision(model)
    return model, processor


def is_degenerate(pred: str) -> bool:
    """Đáp án suy biến do tràn số: một ký tự lặp lại (thường là "!"), dài >= 16. Đếm cho mọi nhánh."""
    s = pred.replace(" ", "").replace("\n", "")
    return len(s) >= 16 and len(set(s)) == 1


def answer(model, processor, images, question: str, suffix: str, max_new_tokens: int) -> str:
    """images: một ảnh PIL hoặc danh sách ảnh (ví dụ ảnh gốc + ảnh crop). Trả lời đã chuẩn hóa NFC."""
    images = images if isinstance(images, list) else [images]
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question + suffix}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)
    inp = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    res = processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return unicodedata.normalize("NFC", res.strip())
