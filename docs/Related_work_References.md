# RELATED WORK & REFERENCES

Danh sách tài liệu cho scope đã chốt: so sánh OCR+LLM, VLM+RAG, các nhánh multimodal (C1, C2) và can thiệp **training-free** vào VLM (ViCrop / HAVC / HAVC với chẩn đoán tiếng Việt). Cập nhật: 2026-10-05.

**Quy ước**
- ★ = bài cốt lõi, nên trích trong Related Work.
- Hội nghị chỉ ghi khi đã kiểm tra được qua trang chính thức hoặc ACL Anthology / proceedings. Bài còn lại ghi *arXiv preprint*.
- Tác giả chỉ ghi khi đã xác minh. Bài khác lấy BibTeX trực tiếp từ arXiv ("Export BibTeX Citation") hoặc Semantic Scholar, **không tự gõ tay**.
- Mới đọc abstract hoặc trang HTML. Cần đọc toàn văn trước khi viết nhận xét chi tiết.
- Một số mã arXiv có tháng gần hiện tại (2606–2609). Mở link xác nhận trước khi nộp.

---

## 1. Kiểm tra code (phục vụ triển khai C3–C5)

| Phương pháp | Code | Model hỗ trợ | Ghi chú cho T4 |
|---|---|---|---|
| ViCrop | [github.com/saccharomycetes/mllms_know](https://github.com/saccharomycetes/mllms_know), MIT | LLaVA-1.5, InstructBLIP, **Qwen2.5-VL** (`qwen2_5_methods.py`, có notebook) | Dùng bản transformers đã sửa đi kèm repo. Biến thể `grad_att` / `pure_grad` cần backward → tốn VRAM; `rel_att` chỉ cần forward. Kết quả Qwen2.5-VL của tác giả ghi "to be released" |
| HAVC | **Không tìm thấy** code, paper không nhắc phát hành | LLaVA-1.5, InstructBLIP (Vicuna-7B) | Phải tự cài đặt theo paper: ngưỡng điểm 0,5, α = 0,4, K = 8 head, τ = 0,1. Có tính gradient lúc inference. Không báo cáo chi phí/thời gian |
| FOCUS | Có trang dự án ([focus-mllm-vqa.github.io](https://focus-mllm-vqa.github.io/)), chưa kiểm tra repo | LLaVA-1.5-7B, LLaVA-OneVision-7B, Qwen2.5-VL-7B | Không cần gradient, ít forward pass hơn ZoomEye 3–6,5 lần. Phương án dự phòng nếu thiếu VRAM |

**Hệ quả cho kế hoạch**
- C3 (ViCrop) chạy được gần như ngay trên Qwen2.5-VL.
- C4 và C5 (HAVC) phải tự cài đặt. Nên dựng trên code ViCrop (cùng cơ chế lấy attention), thêm bước chọn head. Đây là phần việc kỹ thuật lớn nhất, cần đưa vào timeline.
- Vintern chưa được hỗ trợ sẵn, phải tự viết lớp tương tự `qwen2_5_methods.py`.

---

## 2. Danh sách tài liệu theo mục Related Work

### 2.1 Hiểu tài liệu bằng VLM và tài nguyên tiếng Việt
- ★ **Vintern-1B: An Efficient Multimodal Large Language Model for Vietnamese.** arXiv:2408.12480. https://arxiv.org/abs/2408.12480
- ★ **A Survey on Vietnamese Document Analysis and Recognition: Challenges and Future Directions.** arXiv:2506.05061. https://arxiv.org/abs/2506.05061 — nguồn cho vấn đề dấu thanh trong OCR.
- **ViTextVQA** (Vietnamese text-based VQA). arXiv:2404.10652. https://arxiv.org/abs/2404.10652
- **ViOCRVQA** (Vietnamese OCR-VQA). arXiv:2404.18397. https://arxiv.org/abs/2404.18397
- **ViExam: Are Vision Language Models Better than Humans on Vietnamese Multimodal Exam Questions?** arXiv:2508.13680. https://arxiv.org/abs/2508.13680
- **ReceiptVQA** (Vietnamese receipt document VQA). arXiv:2502.19202 *[cần kiểm chứng mã arXiv đúng bài]*. https://arxiv.org/abs/2502.19202
- **ColVintern-1B** (model card, không phải paper). https://huggingface.co/5CD-AI/ColVintern-1B-v1
- Qwen2.5-VL: trích **Qwen2.5-VL Technical Report** *[chưa xác minh mã arXiv, tra trên arXiv]*.

### 2.2 OCR so với VLM trong hỏi đáp / RAG tài liệu
- ★ **OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation.** ICCV 2025. arXiv:2412.02592. https://arxiv.org/abs/2412.02592
- ★ **Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval.** arXiv:2505.05666. https://arxiv.org/abs/2505.05666
- ★ Karamolegkou, A., Angleraud, N., Sagot, B., Clérice, T. **Reading or Guessing? Visual Grounding Failures of Vision-Language Models for OCR in Ancient Greek Editions.** arXiv:2605.27750. https://arxiv.org/abs/2605.27750 — gần nhất về tinh thần: ngôn ngữ ít tài nguyên, VLM so với OCR, đo grounding; can thiệp lúc decode thất bại.

### 2.3 Truy xuất tài liệu dạng ảnh và Multimodal RAG (nhánh B, C1, C2)
- ★ Faysse, M. et al. **ColPali: Efficient Document Retrieval with Vision Language Models.** ICLR 2025. arXiv:2407.01449. https://arxiv.org/abs/2407.01449
- ★ Yu, S. et al. **VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents.** ICLR 2025. arXiv:2410.10594. https://arxiv.org/abs/2410.10594
- **Unifying Multimodal Retrieval via Document Screenshot Embedding (DSE).** EMNLP 2024. arXiv:2406.11251. https://aclanthology.org/2024.emnlp-main.373
- **M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding.** arXiv:2411.04952. https://arxiv.org/abs/2411.04952
- **BGE-M3 (M3-Embedding).** arXiv:2402.03216. https://arxiv.org/abs/2402.03216 — embedding text cho nhánh B.
- **MMDocIR: Benchmarking Multimodal Retrieval for Long Documents.** EMNLP 2025. arXiv:2501.08828. https://arxiv.org/abs/2501.08828
- **ViDoRe V3: A Comprehensive Evaluation of Retrieval Augmented Generation in Complex Real-World Scenarios.** ACL 2026. arXiv:2601.08620. https://aclanthology.org/2026.acl-long.755.pdf
- **UniDoc-Bench: A Unified Benchmark for Document-Centric Multimodal RAG.** arXiv:2510.03663. https://arxiv.org/abs/2510.03663 — so sánh text / image / joint / fusion (đã quyết định không đưa fusion vào thí nghiệm, nhưng vẫn trích trong related work).
- **Comparison of Text-Based and Image-Based Retrieval in Multimodal RAG LLM Systems.** arXiv:2511.16654. https://arxiv.org/abs/2511.16654
- **Evaluating Modern RAG: Textual, Multimodal, Dense, and Late Interaction Pipelines.** arXiv:2608.23176. https://arxiv.org/abs/2608.23176
- **LëtzCross: A Cross-Lingual Page-Level Benchmark for Multimodal Retrieval over Luxembourgish Documents.** arXiv:2608.21714. https://arxiv.org/abs/2608.21714 — mẫu cho nghiên cứu ngôn ngữ ít tài nguyên.
- **M3DR: Towards Universal Multilingual Multimodal Document Retrieval.** arXiv:2512.03514. https://arxiv.org/abs/2512.03514
- Survey: **Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding.** arXiv:2510.15253. https://arxiv.org/abs/2510.15253

### 2.4 Can thiệp training-free vào VLM (trọng tâm, nhánh C3–C5)
- ★ Zhang, J., Khayatkhoei, M., Chhikara, P., Ilievski, F. **MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs (ViCrop).** ICLR 2025. arXiv:2502.17422. https://arxiv.org/abs/2502.17422
- ★ Xie, J., Pan, P., Zhang, X. **Head-Aware Visual Cropping: Enhancing Fine-Grained VQA with Attention-Guided Subimage (HAVC).** arXiv:2601.22483. https://arxiv.org/abs/2601.22483 — bài gần nhất với C5.
- ★ **FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering.** arXiv:2506.21710. https://arxiv.org/abs/2506.21710
- **Paying More Attention to Image: A Training-Free Method for Alleviating Hallucination in LVLMs (PAI).** arXiv:2407.21771 *[hội nghị chưa xác minh]*. https://arxiv.org/abs/2407.21771
- **CropVLM** (chính sách crop học bằng RL; đối chiếu trained với training-free). *[chưa có link paper gốc, tra thêm]*
- **Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings (Light-ColPali).** arXiv:2506.04997. https://arxiv.org/abs/2506.04997
- **PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding.** arXiv:2604.00886. https://arxiv.org/abs/2604.00886
- **Index-Preserving Lightweight Token Pruning for Efficient Document Understanding in VLMs.** ICLR 2026. https://iclr.cc/virtual/2026/10013247

### 2.5 Phân tích layer/head của VLM (cơ sở cho chẩn đoán ở C5)
- ★ **Where Vision Becomes Text: Locating the OCR Routing Bottleneck in Vision-Language Models.** arXiv:2602.22918. https://arxiv.org/abs/2602.22918
- ★ **Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in Visual Document Understanding.** arXiv:2604.04411. https://arxiv.org/abs/2604.04411
- **Vision Function Layer in Multimodal LLMs.** arXiv:2509.24791. https://arxiv.org/abs/2509.24791

### 2.6 Xây dựng dataset và đánh giá
- ★ **Synthetic Document Question Answering in Hungarian.** arXiv:2505.23008. https://arxiv.org/abs/2505.23008 — quy trình sinh + lọc dữ liệu DocVQA cho ngôn ngữ ít tài nguyên.
- **SK-VQA: Synthetic Knowledge Generation at Scale for Training Context-Augmented Multimodal LLMs.** ICML 2025. arXiv:2406.19593. https://arxiv.org/abs/2406.19593
- **Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement.** arXiv:2510.09738. https://arxiv.org/abs/2510.09738
- **Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges.** arXiv:2406.12624. https://arxiv.org/abs/2406.12624
- **Which Works Best for Vietnamese? A Practical Study of Information Retrieval Methods across Domains (ViRE).** Findings of EACL 2026. https://aclanthology.org/2026.findings-eacl.110
- **EvalRAGData** (dataset RAG tiếng Việt 300 mẫu, dataset card). https://huggingface.co/datasets/AITeamVN/EvalRAGData

### 2.7 Cho khóa luận (Agentic RAG, chỉ trích nếu có chương định hướng)
- **ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents.** EMNLP 2025. arXiv:2502.18017. https://aclanthology.org/2025.emnlp-main.464
- **VISOR: Agentic Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning.** arXiv:2604.09508. https://arxiv.org/abs/2604.09508 — có công cụ crop-zoom nhưng huấn luyện bằng RL.
- **Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.** arXiv:2501.09136. https://arxiv.org/abs/2501.09136

---

## 3. Gợi ý khung Related Work (mỗi mục 1–2 đoạn)

1. **VLM cho hiểu tài liệu và tiếng Việt** (2.1): VLM đọc trực tiếp ảnh trang; tài nguyên tiếng Việt chủ yếu là VQA một ảnh (ViTextVQA, ViOCRVQA), chưa có hỏi đáp tài liệu nhiều trang.
2. **OCR so với VLM** (2.2): kết luận chưa thống nhất (OHRBench nghiêng về VLM; Lost in OCR cho thấy OCR bền hơn trên tài liệu lạ); với ngôn ngữ ít tài nguyên, VLM có thể "đoán" thay vì "đọc" (Reading or Guessing).
3. **Multimodal RAG** (2.3): ColPali/VisRAG cho thấy truy xuất bằng ảnh vượt OCR text trên tiếng Anh; LëtzCross và M3DR cho thấy vấn đề với ngôn ngữ ít tài nguyên.
4. **Can thiệp training-free và phân tích nội tại VLM** (2.4, 2.5): ViCrop → HAVC (chọn head bằng tác vụ OCR tiếng Anh) → FOCUS; các bài phân tích layer cho thấy vị trí "đọc chữ" phụ thuộc kiến trúc.
5. **Khoảng trống** (câu cuối của Related Work): chưa có nghiên cứu nào (theo khảo sát) đánh giá can thiệp crop training-free trên tài liệu tiếng Việt có dấu trong pipeline RAG, và chưa kiểm tra liệu tác vụ chẩn đoán head theo ngôn ngữ đích có tốt hơn tác vụ chẩn đoán tiếng Anh hay không.
