# Chẩn đoán head lượt đầu (vision tower fp16), đã thay thế

Giữ lại để so danh sách expert head với lượt chạy lại dùng vision tower fp32 (17/11/2026). Xem results/diag_fp16/.

Kết quả so sánh (17/11/2026): lượt fp32 cho danh sách expert head trùng hoàn toàn với lượt này (EN 35/35, VI 44/44,
cùng thứ tự top-8; Spearman điểm head EN–VI 0,988 → 0,989). Tỉ lệ đọc đúng token VI 99,17% → 99,46%, EN không đổi.
Lỗi tràn số fp16 không làm đổi head được chọn; dùng bản fp32 ở thư mục cha cho mọi nhánh.
