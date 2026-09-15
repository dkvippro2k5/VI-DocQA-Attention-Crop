# Đồ án 2: Baseline pipeline cho hỏi đáp trên tài liệu tiếng Việt

Giai đoạn 1 của đề tài khóa luận tốt nghiệp *"Nghiên cứu Multimodal RAG kết hợp Vision-Language Model cho bài toán hỏi đáp trên tài liệu tiếng Việt"*.

Xem scope chi tiết tại [docs/De_xuat_scope_Do_an_2.md](docs/De_xuat_scope_Do_an_2.md).

## Cấu trúc dự án

```
data/
  raw/          # Tài liệu gốc (PDF, ảnh) — không commit
  processed/    # Ảnh trang đã tách, text OCR — không commit
  qa/           # Dataset Q&A (train/val/test) — commit (nhỏ, dạng json/csv)
src/
  data_prep/    # Tách trang, tiền xử lý tài liệu
  ocr/          # Pipeline OCR (E1)
  vlm/          # Gọi VLM API trực tiếp (E2, E3)
  retrieval/    # Embedding + vector store cho RAG (E4)
  eval/         # EM, F1, LLM-judge, Recall@K, evidence accuracy
notebooks/      # Notebook chạy thử nghiệm trên Colab/Kaggle
results/        # Kết quả từng thí nghiệm E1-E4
docs/           # Tài liệu đề xuất, báo cáo
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # hoặc .venv\Scripts\activate trên Windows
pip install -r requirements.txt
cp .env.example .env  # điền API key
```

## Thí nghiệm

| TN | Mô tả | Thư mục code |
|---|---|---|
| E1 | OCR + LLM | `src/ocr/` |
| E2 | VLM trực tiếp | `src/vlm/` |
| E3 | VLM + Prompt Engineering | `src/vlm/` |
| E4 | VLM + RAG (semantic retrieval) | `src/retrieval/` |
