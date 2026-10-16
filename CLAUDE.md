# CLAUDE.md

Đồ án 2: can thiệp training-free dựa trên attention (crop theo attention) cho VLM trong hỏi đáp tài liệu tiếng Việt, so sánh với OCR+LLM và Multimodal RAG. Scope, RQ và timeline nằm ở [docs/De_xuat_scope_Do_an_2.md](docs/De_xuat_scope_Do_an_2.md); cấu trúc thư mục và bảng nhánh nằm ở [README.md](README.md). Đọc scope trước khi đề xuất thay đổi thí nghiệm.

## Quy tắc nghiên cứu (không được vi phạm)

- **Không huấn luyện / fine-tune** mô hình nào trong ĐA2. Mọi can thiệp chỉ thay đổi inference; trọng số giữ nguyên. LoRA và agent thuộc khóa luận.
- **Tập test chỉ chạy một lần**, sau khi đã chốt cấu hình trên dev. Không chọn nhánh C tốt nhất, layer/head, K, ngưỡng, kích thước crop hay prompt dựa trên kết quả test. Nếu một yêu cầu dẫn tới việc nhìn kết quả test để chỉnh tham số, dừng lại và nói rõ.
- **Chia dev/test theo `document_id`.** Không tài liệu nào xuất hiện ở cả hai. Tài liệu dev/test không bao giờ được dùng để sinh dữ liệu train sau này.
- **Mọi nhánh dùng cùng câu hỏi, cùng prompt sinh đáp án, cùng script đánh giá.** Khi sửa prompt hoặc metric, phải chạy lại tất cả nhánh bị ảnh hưởng.
- **So sánh chính đã khai báo trước là C5 vs C4.** C4 và C5 chỉ khác nhau ở bộ ảnh chẩn đoán (tiếng Anh vs tiếng Việt có dấu); giữ nguyên số ảnh, font, cỡ chữ, siêu tham số.
- Báo cáo cả kết quả âm. Không viết kết luận kiểu "cải thiện" khi chưa qua kiểm định (McNemar / paired bootstrap, Holm cho so sánh phụ).

## Môi trường

- Máy local: Windows, PowerShell; `.venv` ở gốc repo. Code nặng (VLM, retriever, lấy attention) chạy trên **Colab/Kaggle T4 16GB** qua `notebooks/`.
- T4 **không hỗ trợ bf16 gốc và FlashAttention-2**: dùng `torch_dtype=torch.float16`, `attn_implementation="eager"` khi cần attention weights. Luôn giới hạn `max_pixels` cho Qwen2.5-VL.
- HAVC và ViCrop `grad_att` cần backward lúc inference nên tốn VRAM; ưu tiên `rel_att` khi thử nghiệm. Ghi lại VRAM đỉnh (`torch.cuda.max_memory_allocated()`) và thời gian/câu cho mỗi lần chạy.
- API key đọc từ `.env` (`GEMINI_API_KEY`, `OPENAI_API_KEY`) qua `python-dotenv`. Không in, không commit key.

## Code

- ViCrop: dùng code gốc [saccharomycetes/mllms_know](https://github.com/saccharomycetes/mllms_know) (MIT, có `qwen2_5_methods.py`). Vintern chưa được hỗ trợ, cần viết lớp tương tự.
- HAVC không có code công khai; tự cài trong `src/crop/` + `src/diagnosis/`, dựa trên cơ chế lấy attention của ViCrop. Siêu tham số mặc định theo paper: K = 8, ngưỡng 0,5, α = 0,4, τ = 0,1. Mọi chỗ khác paper phải ghi chú trong code và báo cáo.
- Đặt seed cố định; lưu cấu hình (model, max_pixels, K, ngưỡng, layer, prompt) cùng file kết quả trong `results/<nhánh>/` để tái lập được.
- Kết quả theo câu hỏi (không chỉ trung bình) phải được lưu, vì kiểm định ghép cặp cần kết quả từng câu.

## Dữ liệu tiếng Việt

- **Chuẩn hóa Unicode NFC** cho câu hỏi, đáp án, output OCR/VLM trước khi so sánh hoặc tính EM/F1. Dấu thanh có thể ở dạng tổ hợp (NFD) và làm sai metric.
- Không bỏ dấu khi chuẩn hóa; dấu là đối tượng nghiên cứu.
- Trường dữ liệu Q&A: `document_id`, `page_id`, `image`, `question`, `answer`, `evidence_bbox`, `question_type` (`small_detail` / `diacritic` / `table` / `text`), `split`.
- PDF/ảnh gốc và ảnh chẩn đoán không commit (xem `.gitignore`); chỉ commit file Q&A json/csv ở `data/qa/`.

## Tài liệu và trích dẫn

- Viết tài liệu, báo cáo, comment giải thích bằng **tiếng Việt**; tên biến, hàm, file bằng tiếng Anh.
- Danh sách tài liệu: [docs/Related_work_References.md](docs/Related_work_References.md). Không tự gõ BibTeX hay tên tác giả từ trí nhớ; lấy từ arXiv/Semantic Scholar. Mục đánh dấu *[cần kiểm chứng]* chưa được xác minh.
- Khi viết về một paper, phân biệt rõ đã đọc toàn văn hay chỉ đọc abstract.

## Git

- Branch mặc định local là `master`; không commit hoặc push khi chưa được yêu cầu.
