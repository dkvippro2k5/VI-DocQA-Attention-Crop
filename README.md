# Đồ án 2: Can thiệp dựa trên attention cho VLM khi đọc trang tài liệu tiếng Việt

Crop training-free theo attention (ViCrop, HAVC, và HAVC với bước chẩn đoán head bằng chữ tiếng Việt có dấu), so với LoRA nhắm vào các head đã chẩn đoán. Đánh giá theo thiết kế 2 × 2: ngôn ngữ (Anh / Việt) × loại ảnh (ảnh cảnh / tài liệu dày chữ), ở chế độ trang đúng.

Giai đoạn 1 của khóa luận tốt nghiệp. Truy xuất trên tài liệu nhiều trang (OCR+LLM, RAG văn bản, Multimodal RAG) và agent để dành cho khóa luận (đổi ngày 15/11/2026).

Xem scope chi tiết tại [docs/De_xuat_scope_Do_an_2.md](docs/De_xuat_scope_Do_an_2.md) và danh sách tài liệu tại [docs/Related_work_References.md](docs/Related_work_References.md).

## Cấu trúc dự án

```
data/
  raw/          # Tài liệu gốc (PDF, ảnh) và bộ dữ liệu tải về (ViInfographicVQA, ...) — không commit
  processed/    # Ảnh trang đã tách, text OCR — không commit
  qa/
    dev/        # Tập dev: tinh chỉnh siêu tham số — commit (json/csv)
    test/       # Tập test: chạy một lần với cấu hình đã chốt — commit (json/csv)
  diagnostic/
    en/         # ~1.000 ảnh chữ tiếng Anh chẩn đoán head cho C4 — không commit
    vi/         # ~1.000 ảnh chữ tiếng Việt có dấu tự render cho C5 — không commit
src/
  data_prep/    # Tách trang, tiền xử lý tài liệu, quy trình gán nhãn
  vlm/          # Chạy VLM: Qwen2.5-VL-3B, Vintern-1B; prompt dùng chung
  crop/         # Can thiệp crop theo attention: ViCrop (C3), HAVC (C4, C5), crop ngẫu nhiên (R)
  diagnosis/    # Sinh ảnh chẩn đoán, chọn head (K, ngưỡng, entropy, gradient)
  lora/         # LoRA nhắm head (C6) và LoRA toàn bộ (C7)
  eval/         # EM, F1, LLM-judge, IoU, McNemar, paired bootstrap, Holm
  ocr/          # OCR tiếng Việt (dùng cho gán nhãn; nhánh A thuộc khóa luận)
  retrieval/    # Retriever (thuộc khóa luận)
notebooks/      # Notebook chạy thử nghiệm trên Colab/Kaggle
results/        # Kết quả từng nhánh thí nghiệm
docs/           # Tài liệu đề xuất, báo cáo
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # hoặc .venv\Scripts\activate trên Windows
pip install -r requirements.txt
cp .env.example .env  # điền API key
```

## Các nhánh thí nghiệm

Tất cả nhánh chạy trên cùng tập test, cùng câu hỏi, cùng prompt sinh đáp án và cùng script chấm. Đầu vào là trang chứa đáp án.

| Nhánh | Nội dung | Thư mục code | Thư mục kết quả |
|---|---|---|---|
| B0 | Model gốc (Qwen2.5-VL-3B; Vintern-1B tham khảo) | `src/vlm/` | `results/B0_base/` |
| B1 | Tiền xử lý: tăng `max_pixels`, xử lý ảnh | `src/vlm/` | `results/B1_preproc/` |
| C3 | ViCrop | `src/crop/` | `results/C3_vicrop/` |
| C4 | HAVC (head chọn bằng chữ tiếng Anh) | `src/crop/`, `src/diagnosis/` | `results/C4_havc_en/` |
| C5 | HAVC với chẩn đoán tiếng Việt có dấu | `src/crop/`, `src/diagnosis/` | `results/C5_havc_vi/` |
| R | Crop ngẫu nhiên cùng kích thước với C5 | `src/crop/` | `results/R_random_crop/` |
| C6 | LoRA nhắm các head đã chẩn đoán | `src/lora/` | `results/C6_lora_heads/` |
| C7 | LoRA toàn bộ attention (đối chứng C6) | `src/lora/` | `results/C7_lora_full/` |

So sánh chính khai báo trước: C5 vs C4. Thư mục `results/A_*`, `B_*`, `C1_*`, `C2_*` giữ lại cho phần truy xuất ở khóa luận.
