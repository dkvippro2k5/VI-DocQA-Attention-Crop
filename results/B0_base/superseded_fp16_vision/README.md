# Lượt B0 dev đầu tiên (17/11/2026), đã thay thế

Chạy với vision tower fp16. 6/409 câu bị tràn số ở `visual.blocks.31.mlp.down_proj` và sinh chuỗi "!!!!…"
(xem `results/diag_fp16/`, `notebooks/fp16_overflow_diag.ipynb`). Từ 17/11/2026 mọi nhánh chạy vision tower fp32
(`src/vlm/qwen.py`). Giữ lại để đối chiếu, không dùng để so sánh nhánh.
