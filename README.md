# Đồ án 2: Can thiệp training-free dựa trên attention cho VLM trong hỏi đáp tài liệu tiếng Việt

So sánh OCR+LLM, VLM+RAG văn bản và Multimodal RAG; đề xuất và kiểm chứng biến thể crop theo attention (HAVC) với bước chẩn đoán head bằng chữ tiếng Việt có dấu.

Giai đoạn 1 của khóa luận tốt nghiệp. Đồ án 2 không huấn luyện mô hình; LoRA nhắm vào layer/head đã chẩn đoán và agent để dành cho khóa luận.

Xem scope chi tiết tại [docs/De_xuat_scope_Do_an_2.md](docs/De_xuat_scope_Do_an_2.md) và danh sách tài liệu tại [docs/Related_work_References.md](docs/Related_work_References.md).

## Cấu trúc dự án

```
data/
  raw/          # Tài liệu gốc (PDF, ảnh) — không commit
  processed/    # Ảnh trang đã tách, text OCR — không commit
  qa/
    dev/        # Tập dev: chọn nhánh C tốt nhất, tinh chỉnh siêu tham số — commit (json/csv)
    test/       # Tập test: chạy một lần với cấu hình đã chốt — commit (json/csv)
  diagnostic/
    en/         # ~1.000 ảnh chữ tiếng Anh (Synthdog-en) chẩn đoán head cho C4 — không commit
    vi/         # ~1.000 ảnh chữ tiếng Việt có dấu tự render cho C5 — không commit
src/
  data_prep/    # Tách trang, tiền xử lý tài liệu, sinh nháp Q&A
  ocr/          # OCR tiếng Việt (nhánh A, B)
  vlm/          # Chạy VLM: Vintern-1B, Qwen2.5-VL-3B (nhánh B, C1–C5)
  retrieval/    # Retriever văn bản (BGE-M3 + FAISS) và retriever ảnh (ColVintern, ColQwen)
  crop/         # Can thiệp crop theo attention: ViCrop (C3), HAVC (C4, C5)
  diagnosis/    # Sinh ảnh chẩn đoán, chọn head (K, ngưỡng, entropy, gradient)
  eval/         # EM, F1, LLM-judge, Recall@K, MRR, IoU, McNemar, paired bootstrap
notebooks/      # Notebook chạy thử nghiệm trên Colab/Kaggle
results/        # Kết quả từng nhánh thí nghiệm (A, B, C1–C5)
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

Tất cả nhánh chạy trên cùng tập test, cùng câu hỏi và cùng prompt sinh đáp án.

| Nhánh | Pipeline | Thư mục code | Thư mục kết quả |
|---|---|---|---|
| A | OCR + retrieval văn bản + LLM | `src/ocr/`, `src/retrieval/` | `results/A_ocr_text_rag_llm/` |
| B | Retrieval văn bản (trên OCR) + VLM đọc ảnh trang | `src/retrieval/`, `src/vlm/` | `results/B_text_rag_vlm/` |
| C1 | ColVintern-1B + Vintern-1B | `src/retrieval/`, `src/vlm/` | `results/C1_colvintern_vintern/` |
| C2 | ColQwen + Qwen2.5-VL-3B | `src/retrieval/`, `src/vlm/` | `results/C2_colqwen_qwen25vl/` |
| C3 | Nhánh C tốt nhất + ViCrop | `src/crop/` | `results/C3_vicrop/` |
| C4 | Nhánh C tốt nhất + HAVC (head chọn bằng chữ tiếng Anh) | `src/crop/`, `src/diagnosis/` | `results/C4_havc_en/` |
| C5 | Nhánh C tốt nhất + HAVC với chẩn đoán tiếng Việt có dấu | `src/crop/`, `src/diagnosis/` | `results/C5_havc_vi/` |

"Nhánh C tốt nhất" (C1 hoặc C2) được chọn trên tập dev, không chọn trên tập test.
