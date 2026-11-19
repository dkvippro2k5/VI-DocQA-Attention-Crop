# Lượt B0 dev đầu tiên (17/11/2026), đã thay thế

Chạy với vision tower fp16. 6/409 câu bị tràn số ở `visual.blocks.31.mlp.down_proj` và sinh chuỗi "!!!!…"
(xem `results/diag_fp16/`, `notebooks/fp16_overflow_diag.ipynb`). Từ 17/11/2026 mọi nhánh chạy vision tower fp32
(`src/vlm/qwen.py`). Giữ lại để đối chiếu, không dùng để so sánh nhánh.

## Đối chiếu với lượt vision fp32 (18/11/2026, `results/B0_base/dev_*.jsonl`)

Cùng 300 câu dev, cùng prompt p2, greedy. EM / F1 tính lấy điểm cao nhất trên các đáp án chấp nhận.

| Bộ | EM fp16 → fp32 | F1 fp16 → fp32 | Suy biến "!!!!" | Đáp án giống hệt | EM đổi |
|---|---|---|---|---|---|
| TextVQA | 63,0 → 65,0 | 73,2 → 76,1 | 2 → 0 | 96/100 | +2 / −0 |
| DocVQA | 81,0 → 81,0 | 86,9 → 86,9 | 0 → 0 | 100/100 | 0 |
| ViTextVQA | 25,0 → 25,0 | 51,5 → 51,6 | 4 → 0 | 91/100 | +1 / −1 |

Cả 6 câu suy biến đều hết; 2 trong số đó thành đúng EM (textvqa_38125, vitextvqa_38079). Ngoài các câu suy biến,
5 câu đổi đáp án do sai khác số học nhỏ (1 câu thành đúng, 1 câu thành sai: vitextvqa_17990). Chi phí: 2,45 / 2,76 /
2,62 s/câu (fp16: 2,11 / 1,69 / 2,81); VRAM đỉnh 7,4 GiB trên GPU 0 và 3,2 GiB trên GPU 1 (T4 x2, device_map auto).
