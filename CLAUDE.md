# CLAUDE.md

Đồ án 2: can thiệp dựa trên attention cho VLM khi đọc một trang tài liệu tiếng Việt. Gồm crop training-free (ViCrop, HAVC, HAVC chẩn đoán bằng chữ có dấu) và LoRA nhắm vào các head đã chẩn đoán, đánh giá theo thiết kế 2 × 2 (Anh/Việt × ảnh cảnh/tài liệu). Truy xuất nhiều trang (OCR+LLM, RAG, ColQwen/ColVintern) và agent thuộc khóa luận (đổi ngày 15/11/2026, GVHD đã đồng ý). Scope, RQ và timeline nằm ở [docs/De_xuat_scope_Do_an_2.md](docs/De_xuat_scope_Do_an_2.md); cấu trúc thư mục và bảng nhánh nằm ở [README.md](README.md). Đọc scope trước khi đề xuất thay đổi thí nghiệm.

## Quy tắc nghiên cứu (không được vi phạm)

- **Chỉ nhánh C6, C7 được huấn luyện**, và chỉ dưới dạng LoRA/QLoRA (trọng số gốc giữ nguyên). Mọi nhánh khác chỉ thay đổi inference.
- **Dữ liệu train chỉ lấy từ bộ công khai có người viết hoặc người duyệt** (ViInfographicVQA, ViTextVQA train, MTVQA-VI train; xem scope mục 7.2). Không dùng nhãn LLM chưa duyệt. Bộ nào chỉ có nhãn LLM (ví dụ Viet-Doc-VQA-II) thì phải kiểm 100 mẫu và qua ngưỡng đặt trước mới được dùng.
- **Tập test chỉ chạy một lần**, sau khi đã chốt cấu hình trên dev. Không chọn layer/head, K, ngưỡng, kích thước crop, prompt, rank/learning rate/số bước LoRA dựa trên kết quả test. Nếu một yêu cầu dẫn tới việc nhìn kết quả test để chỉnh tham số, dừng lại và nói rõ.
- **Chia dev/test theo `document_id`.** Không tài liệu nào xuất hiện ở cả hai. Tài liệu dev/test (và ảnh dev/test của ba bộ công khai trong thiết kế 2 × 2) không bao giờ được dùng để train.
- **Mọi nhánh dùng cùng câu hỏi, cùng prompt sinh đáp án, cùng script đánh giá.** Khi sửa prompt hoặc metric, phải chạy lại tất cả nhánh bị ảnh hưởng.
- **So sánh chính đã khai báo trước là C5 vs C4.** C4 và C5 chỉ khác nhau ở bộ ảnh chẩn đoán (tiếng Anh vs tiếng Việt có dấu); giữ nguyên số ảnh, font, cỡ chữ, siêu tham số. C6 và C7 chỉ khác nhau ở chỗ gắn LoRA; giữ nguyên dữ liệu, số bước, siêu tham số.
- Báo cáo cả kết quả âm. Không viết kết luận kiểu "cải thiện" khi chưa qua kiểm định (McNemar / paired bootstrap, Holm cho so sánh phụ).

## Môi trường

- Máy local: Windows, PowerShell; `.venv` ở gốc repo. Code nặng (VLM, retriever, lấy attention) chạy trên **Colab/Kaggle T4 16GB** qua `notebooks/`.
- T4 **không hỗ trợ bf16 gốc và FlashAttention-2**: dùng `torch_dtype=torch.float16`, `attn_implementation="eager"` khi cần attention weights. Luôn giới hạn `max_pixels` cho Qwen2.5-VL.
- **Vision tower của Qwen2.5-VL chạy fp32** (LLM vẫn fp16), qua `src/vlm/qwen.py` (`vision_fp32=True`). Vision tower fp16 bị tràn số ở `visual.blocks.31.mlp.down_proj` và sinh chuỗi "!!!!…" (xem `notebooks/fp16_overflow_diag.ipynb`). Kết quả chạy trước 17/11/2026 với vision fp16 nằm trong `superseded_fp16_vision/`, không dùng để so sánh nhánh.
- HAVC và ViCrop `grad_att` cần backward lúc inference nên tốn VRAM; ưu tiên `rel_att` khi thử nghiệm. Ghi lại VRAM đỉnh (`torch.cuda.max_memory_allocated()`) và thời gian/câu cho mỗi lần chạy.
- Train LoRA: fp16 (không bf16), gradient checkpointing, giới hạn `max_pixels`; không cần eager. Ghi lại VRAM đỉnh, thời gian train, số tham số train. Hạn mức Kaggle khoảng 30 giờ GPU/tuần.
- API key đọc từ `.env` (`GEMINI_API_KEY`, `OPENAI_API_KEY`) qua `python-dotenv`. Không in, không commit key.

## Code

- ViCrop: dùng code gốc [saccharomycetes/mllms_know](https://github.com/saccharomycetes/mllms_know) (MIT, có `qwen2_5_methods.py`). Vintern chưa được hỗ trợ, cần viết lớp tương tự.
- HAVC không có code công khai; tự cài trong `src/crop/` + `src/diagnosis/`, dựa trên cơ chế lấy attention của ViCrop. Siêu tham số mặc định theo paper: K = 8, ngưỡng 0,5, α = 0,4, τ = 0,1. Mọi chỗ khác paper phải ghi chú trong code và báo cáo.
- Đặt seed cố định; lưu cấu hình (model, max_pixels, K, ngưỡng, layer, prompt) cùng file kết quả trong `results/<nhánh>/` để tái lập được.
- Kết quả theo câu hỏi (không chỉ trung bình) phải được lưu, vì kiểm định ghép cặp cần kết quả từng câu.
- LoRA (C6, C7) đặt trong `src/lora/`; lưu cấu hình train (dữ liệu, số mẫu, rank, alpha, learning rate, số bước, các module gắn LoRA, danh sách head) cùng kết quả.

## Dữ liệu tiếng Việt

- **Chuẩn hóa Unicode NFC** cho câu hỏi, đáp án, output OCR/VLM trước khi so sánh hoặc tính EM/F1. Dấu thanh có thể ở dạng tổ hợp (NFD) và làm sai metric.
- Không bỏ dấu khi chuẩn hóa; dấu là đối tượng nghiên cứu.
- Trường dữ liệu Q&A: `document_id`, `page_id`, `image`, `question`, `answer`, `evidence_bbox`, `question_type` (`small_detail` / `table` / `text`, gán bằng luật ở scope mục 9), `has_diacritic`, `split`.
- PDF/ảnh gốc, ảnh chẩn đoán và bộ dữ liệu tải về (`data/raw/`) không commit (xem `.gitignore`); chỉ commit file Q&A json/csv ở `data/qa/`.
- Không bỏ dấu khi chuẩn hóa không có nghĩa là không chấm bỏ dấu: điểm bỏ dấu được báo cáo thêm như một phép phân tích lỗi do dấu, không thay chỉ số chính.

## Tài liệu và trích dẫn

- Viết tài liệu, báo cáo, comment giải thích bằng **tiếng Việt**; tên biến, hàm, file bằng tiếng Anh.
- Danh sách tài liệu: [docs/Related_work_References.md](docs/Related_work_References.md). Không tự gõ BibTeX hay tên tác giả từ trí nhớ; lấy từ arXiv/Semantic Scholar. Mục đánh dấu *[cần kiểm chứng]* chưa được xác minh.
- Khi viết về một paper, phân biệt rõ đã đọc toàn văn hay chỉ đọc abstract.

## Git

- Branch mặc định là `main`; commit thẳng vào `main`. Không commit hoặc push khi chưa được yêu cầu.
