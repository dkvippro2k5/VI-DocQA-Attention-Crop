# C3 (ViCrop rel_att) trên dev cũ, đã thay thế

Lượt chạy tuần 3–4 (commit 50d0c0f, ac6ae71), trước khi đổi scope và bộ nhãn:
- prompt p1 ("Trả lời ngắn gọn bằng một cụm từ hoặc một câu.", `max_new_tokens` = 64);
- vision tower fp16 (trước khi chuyển sang fp32 ngày 17/11/2026);
- tập dev đầu tiên (137 câu VisR-Bench, qid dạng `_q1`), không còn dùng sau khi duyệt lại và chia dev/test theo tài liệu.

Gồm kết quả ảnh nguyên / crop (`dev_raw.jsonl`) và quét layer × kích thước hộp (`dev_sweep.jsonl`, layer 14/22/30,
hộp 336–1024). Chỉ giữ để tham khảo, **không dùng để so sánh nhánh hay chọn cấu hình**: mọi nhánh chạy lại với
prompt p2, vision fp32 trên dev mới (scope mục 9).
