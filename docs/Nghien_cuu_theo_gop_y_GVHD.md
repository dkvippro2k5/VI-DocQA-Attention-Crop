# NGHIÊN CỨU THEO GÓP Ý CỦA GVHD

Tài liệu trả lời 6 góp ý của GVHD về đề xuất scope ĐA2 ([De_xuat_scope_Do_an_2.md](De_xuat_scope_Do_an_2.md)). Ngày: 2026-10-02.

> **Lưu ý về độ tin cậy:** thông tin về paper/benchmark lấy từ tìm kiếm web (có link). Các con số phần cứng và mô tả kiến trúc có chỗ dựa trên kiến thức chung, đánh dấu **[cần kiểm chứng]** — hãy tự đo/đọc lại paper gốc trước khi đưa vào báo cáo.

---

## 1. Paper liên quan và các giải pháp đã có

Pipeline của ĐA2: OCR+LLM (E1) → VLM trực tiếp (E2) → +prompt (E3) → +RAG text (E4) → +Multimodal RAG (E5) → (khóa luận) Agentic RAG (E6), LoRA (E7).

### 1.1 Retrieval trên tài liệu dạng ảnh (trọng tâm E4/E5)

| Paper | Ý chính | Liên quan tới ĐA2 |
|---|---|---|
| [ColPali (ICLR 2025)](https://arxiv.org/html/2407.01449v5) + benchmark ViDoRe | VLM sinh multi-vector cho từng patch ảnh trang, so khớp late-interaction (MaxSim) với token câu hỏi; bỏ hẳn OCR/layout parsing | Nền tảng của E5 cấp trang. Baseline chính để so với text retrieval |
| [DSE – Document Screenshot Embedding (EMNLP 2024)](https://arxiv.org/html/2406.11251v2) | Mã hóa ảnh chụp trang thành **1 vector** (Phi-3-vision); hơn BM25 17 điểm top-1, hơn retrieval trên OCR text >15 nDCG@10 ở slide | Phương án rẻ hơn ColPali (index nhỏ, dùng FAISS thường) |
| [VisRAG (ICLR 2025)](https://arxiv.org/html/2410.10594v2) | Pipeline RAG hoàn toàn bằng VLM: embed trang như ảnh, sinh đáp án bằng VLM; báo cáo +20–40% end-to-end so với text RAG | **Gần như đúng là E4 vs E5** → dùng làm paper đối chiếu chính cho RQ3 |
| [M3DocRAG](https://arxiv.org/html/2411.04952v1) | Multimodal RAG đa trang, đa tài liệu (ColPali + VLM), benchmark M3DocVQA (3.000+ PDF, 40.000+ trang) | Thiết kế open-domain/multi-hop cho khóa luận |
| [MMDocIR](https://arxiv.org/html/2501.08828v3) | Benchmark retrieval **cấp trang và cấp layout** (đoạn, bảng, hình, công thức); 313 tài liệu, 1.658 query có nhãn chuyên gia | **Mẫu thiết kế evidence cấp vùng** cho E5 (đúng thứ bạn đang muốn gán nhãn) |
| [ViDoRe V3 (ACL 2026)](https://arxiv.org/html/2601.08620v1) | 10 dataset, 26k trang, 3.099 query người xác minh, có bounding box + đáp án chuẩn; visual retriever > text retriever; late-interaction + text reranker cải thiện mạnh; mô hình vẫn yếu ở phần phi văn bản và grounding mịn | Giao thức gán nhãn tham khảo; kết luận của nó là giả thuyết để bạn kiểm chứng trên tiếng Việt |
| [ViDoRAG (EMNLP 2025)](https://arxiv.org/html/2502.18017v2) | Hybrid retrieval (GMM) + 3 agent Seeker/Inspector/Answer lặp có phản hồi; benchmark ViDoSeek | Mẫu kiến trúc cho E6 |

### 1.2 Survey (đọc đầu tiên cho chương Tổng quan)

- [Scaling Beyond Context: Survey of Multimodal RAG for Document Understanding](https://arxiv.org/html/2510.15253v2) — taxonomy theo domain / modality retrieval / granularity (đúng trục ĐA2 đề xuất).
- [Agentic RAG: A Survey](https://arxiv.org/pdf/2501.09136) và [SoK: Agentic RAG](https://arxiv.org/pdf/2603.07379) — dùng cho phần E6 của khóa luận (planning, tool use, reflection).

### 1.3 Tiếng Việt: những gì đã có

| Tài nguyên | Nội dung | Khoảng trống |
|---|---|---|
| [Vintern-1B](https://arxiv.org/html/2408.12480v1) | VLM 1B (InternViT-300M + Qwen2-0.5B), tinh chỉnh >3 triệu cặp ảnh-hỏi-đáp tiếng Việt (OCR, DocVQA) | Chạy được trên T4, ứng viên VLM open-source cho E2 |
| [ColVintern-1B](https://huggingface.co/5CD-AI/ColVintern-1B-v1/raw/33a156a229e094f8223ad190ac0498b67aac483f/README.md) | ColPali-style retriever tiếng Việt/Anh trên Vintern-1B | **Ứng viên chính cho E5**; chưa thấy đánh giá độc lập trên tài liệu tiếng Việt |
| [ViTextVQA](https://arxiv.org/pdf/2404.10652v5), [ViOCRVQA](https://www.alphaxiv.org/benchmarks/vietnam-national-university/viocrvqa), ReceiptVQA | VQA trên ảnh có chữ (biển hiệu, bìa sách, hóa đơn) | Đều là **1 ảnh – 1 câu hỏi**, không có kho đa trang, không có retrieval, không có evidence cấp vùng |
| [ViExam / VMMU](https://arxiv.org/html/2508.13680v2) | Câu hỏi đa phương thức tiếng Việt (đề thi, biểu đồ, bảng) | Đo khả năng suy luận của VLM, không phải RAG trên tài liệu |
| [Survey Vietnamese Document Analysis & Recognition](https://arxiv.org/pdf/2506.05061) | Tổng quan OCR/DAR tiếng Việt | Đọc cho phần OCR (E1) |
| Embedding văn bản: [BGE-M3](https://huggingface.co/nrl-ai/vietnamese-embedding-bge-m3) (đa ngôn ngữ, 8k token, dense+sparse+multi-vector) và các bản fine-tune tiếng Việt | Retriever text cho E4 | — |

**Kết luận khảo sát:** theo tìm kiếm của tôi, chưa có benchmark **Multimodal RAG trên tài liệu tiếng Việt** có kho đa trang + evidence cấp trang/vùng. ViDoRe V3 có 6 ngôn ngữ nhưng dường như không gồm tiếng Việt **[cần kiểm chứng trên trang benchmark]**. Đây là khoảng trống có thể làm thành đóng góp.

---

## 2. Tìm kiếm / scale dataset

### 2.1 Nguồn dữ liệu (tài liệu thật, tiếng Việt)

- Văn bản hành chính: công báo, cổng thông tin chính phủ (thông tư, nghị định, quyết định) — công khai, nhiều trang, có bảng.
- Đề thi/đề cương: THPT quốc gia, đề thi đại học công khai (có hình, công thức, bảng).
- Báo cáo thường niên / báo cáo tài chính doanh nghiệp niêm yết, tài liệu giáo trình mở.
- **Kiểm tra giấy phép/điều khoản sử dụng** trước khi phát hành dataset công khai.

### 2.2 Cách scale câu hỏi mà không tốn hết công annotation

Cách làm của [Synthetic Document QA in Hungarian](https://arxiv.org/pdf/2505.23008) rất sát tình huống của bạn: nguồn tài liệu → LLM sinh câu hỏi/đáp án → **lọc chất lượng** → tinh chỉnh gain tới +7.2% trên VLM mạnh. MMDocIR cũng chia 2 mức: 1.685 câu **chuyên gia gán** (test) + 173.843 câu **bootstrap** (train).

Đề xuất quy trình 3 bậc:

1. **LLM/VLM sinh nháp**: đưa ảnh trang (hoặc crop bảng/hình) vào VLM mạnh (Gemini/GPT/Claude) → sinh câu hỏi + đáp án + vùng bằng chứng, kèm loại câu hỏi.
2. **Lọc tự động**: bỏ câu hỏi trả lời được mà không cần ảnh (kiểm tra bằng LLM text-only không có tài liệu), bỏ câu mơ hồ ("trang này nói gì"), bỏ trùng lặp, câu hỏi phải tự đủ nghĩa khi đứng độc lập (vì dùng cho retrieval trên kho nhiều trang).
3. **Người duyệt**: tập **test** do người kiểm 100% (sửa đáp án, vẽ lại bbox); tập train/dev có thể chỉ duyệt mẫu.

Mục tiêu số lượng: **test ≥ 300 câu người duyệt** (xem mục 6 về lý do), train/dev synthetic vài nghìn nếu khóa luận làm fine-tuning.

---

## 3. Có nên scale lên khóa luận? Đóng góp nghiên cứu và phần có thể bóc tách

### 3.1 Trả lời ngắn

**Có**, nhưng không nên coi "scale" là đóng góp. Đóng góp nghiên cứu cần là một **câu hỏi có câu trả lời mới**. Với khoảng trống ở mục 1.3, có 4 hướng, xếp theo độ khả thi/giá trị:

| # | Hướng đóng góp | Câu hỏi nghiên cứu | Khả thi trên T4 | Ghi chú |
|---|---|---|---|---|
| A | **Benchmark Multimodal RAG tiếng Việt** (kho đa trang, evidence trang + layout, có nhãn loại câu hỏi) | Retriever/VLM hiện nay yếu ở đâu trên tiếng Việt (dấu, bảng, đề thi)? | Cao (chủ yếu annotate + gọi API) | Đóng góp tài nguyên, publish được; đúng việc ĐA2 đã làm |
| B | **Fine-tune retriever** (ColVintern/ColQwen-nhỏ) bằng dữ liệu synthetic tiếng Việt | Dữ liệu synthetic có nâng Recall@K so với model multilingual có sẵn không? | Trung bình–cao (retriever 1–3B + LoRA, T4/Kaggle) | **Rẻ hơn nhiều so với fine-tune VLM sinh đáp án, hiệu quả dễ đo** (Recall@K) |
| C | **Fine-tune VLM sinh đáp án (LoRA/QLoRA)** | LoRA có giảm hallucination/cải thiện EM trên tiếng Việt không? | Trung bình (VLM 2–3B hoặc Vintern-1B; 7B rất sát giới hạn) | Đã nằm trong E7 |
| D | **Agentic RAG** (E6) | Multi-hop + self-verification có giảm hallucination so với single-shot? | Cao (chủ yếu gọi API, LangGraph) | Đúng mục tiêu hồ sơ AI Engineer đã nêu trong tài liệu scope |

### 3.2 Ý "chạy trên tài liệu tiếng Anh rồi dịch sang tiếng Việt"

Tôi hiểu ý GVHD là: lấy tài liệu/dataset tiếng Anh sẵn có (ViDoRe, MMDocIR, DocVQA, M3DocVQA) → chuyển sang tiếng Việt → dùng để fine-tune hoặc nghiên cứu. **[Nếu GVHD ý khác, cần hỏi lại.]** Có thể khai thác theo 3 mức:

1. **Dịch Q&A, giữ ảnh gốc tiếng Anh** — dễ nhất, nhưng tạo ra cặp *câu hỏi tiếng Việt – tài liệu tiếng Anh* (cross-lingual retrieval). Đây là bài toán khác, thực tế hợp lý (người Việt hỏi tiếng Việt trên tài liệu quốc tế) và đo được khả năng cross-lingual của ColPali/BGE-M3. Rủi ro: "translationese" như đã nêu trong [bài Hungarian](https://arxiv.org/pdf/2505.23008).
2. **Dịch rồi render lại tài liệu** (trích text → dịch → dựng lại PDF/HTML tiếng Việt) — khó giữ nguyên layout/bảng/hình, mất công, chất lượng không đều.
3. **Sinh tài liệu tiếng Việt tổng hợp** (LLM viết nội dung → dựng bằng HTML/LaTeX template có bảng, biểu đồ) — kiểm soát được layout và nhãn bbox *miễn phí* (vì biết tọa độ khi render). Cách này tương tự nhóm HuCCPDF trong bài Hungarian. Nhược điểm: gap giữa tài liệu tổng hợp và tài liệu thật → vẫn cần test set thật do người gán.

**Khuyến nghị:** dùng (3) hoặc (1) cho **train**, giữ test là tài liệu **thật** tiếng Việt. Nghiên cứu được: *"fine-tune bằng dữ liệu dịch/tổng hợp có chuyển giao sang tài liệu thật không?"* — đây là một RQ rõ ràng và có số liệu kiểm chứng.

### 3.3 Phần có thể bóc tách thành đóng góp độc lập

- Dataset + giao thức gán nhãn (A) — bóc riêng được, không phụ thuộc agent.
- Retriever fine-tuning (B) — bóc riêng, đo bằng Recall@K, không cần LLM-judge.
- Phân tích lỗi theo loại câu hỏi (text / bảng / hình / công thức) + theo chất lượng OCR — một chương phân tích có giá trị dù kết quả âm.
- Agent (D) — bóc riêng, chạy trên dataset (A) đã có.

Nên **chốt 1 đóng góp chính + 1 phụ** cho khóa luận. Gợi ý: chính = (A)+(D) như scope đã định; phụ = (B) nếu còn thời gian (rẻ hơn E7 mà dễ có kết quả định lượng hơn).

---

## 4. Yêu cầu phần cứng

Số liệu bên dưới là **ước lượng** ([nguồn tham khảo](https://willitrunai.com/de/can-run/qwen-2.5-vl-7b-on-t4-16gb)); hãy đo thực tế bằng `torch.cuda.max_memory_allocated()` **[cần kiểm chứng]**.

| Tác vụ | Mô hình ví dụ | VRAM ước lượng | T4 16GB |
|---|---|---|---|
| Inference VLM FP16 | Qwen2.5-VL-7B | ~18 GB | ❌ không vừa |
| Inference VLM 4-bit | Qwen2.5-VL-7B | ~5–8 GB (tăng theo độ phân giải ảnh) | ✅ |
| Inference VLM nhỏ | Vintern-1B, Qwen2.5-VL-3B | 2–7 GB | ✅ |
| Indexing ColPali/ColVintern | 1–3B | vài GB | ✅ (chậm) |
| QLoRA VLM 2–3B | Qwen2.5-VL-3B, Vintern-1B | ~10–16 GB tùy độ dài chuỗi/ảnh | ⚠️ sát giới hạn, giảm độ phân giải ảnh |
| QLoRA VLM 7B | Qwen2.5-VL-7B | thường >16 GB với ảnh độ phân giải cao | ❌/rất sát |
| Full fine-tune 7B | — | ~4× inference | ❌ |

Điểm cần nhớ khi dùng T4:
- T4 **không hỗ trợ bfloat16 gốc và FlashAttention-2** → dùng fp16; một số model/mã ví dụ mặc định bf16 sẽ lỗi hoặc chậm **[cần kiểm chứng với từng model]**.
- **Số token ảnh quyết định VRAM**: Qwen2.5-VL dùng độ phân giải động → trang A4 độ phân giải cao tạo hàng nghìn token. Đặt `max_pixels` để kiểm soát.
- **Dung lượng index ColPali**: khoảng ~1.030 vector × 128 chiều mỗi trang (theo bài ColPali) ≈ 260 KB/trang ở fp16 → 1.000 trang ≈ 260 MB. Không là vấn đề với kho cỡ ĐA2.
- Colab/Kaggle free có **giới hạn thời gian phiên và quota/tuần** (Kaggle thường 2×T4, ~30 GB tổng nhưng không gộp tự động) → lưu checkpoint, chạy theo lô.
- Phương án dự phòng đã có trong scope: dùng API cho VLM sinh đáp án + LLM-judge; chỉ chạy **local** phần embedding/retrieval. Nên ghi lại **chi phí API/câu hỏi** để báo cáo.

---

## 5. Hiểu sâu phía sau từng model — danh sách cần nắm

Mức mục tiêu: giải thích được vì sao kết quả như vậy (phục vụ phân tích lỗi và phản biện), không cần chứng minh toán học.

| Thành phần | Cần hiểu | Đọc | Tự kiểm bằng thực nghiệm nhỏ |
|---|---|---|---|
| **OCR (E1)** | Hai pha detect + recognize; lỗi dấu tiếng Việt; mất cấu trúc bảng/thứ tự đọc | [Survey Vietnamese DAR](https://arxiv.org/pdf/2506.05061) | Đo CER/WER trên 20 trang; xem lỗi dấu nào phổ biến |
| **VLM (E2)** | Vision encoder (ViT) → projector → LLM; ảnh thành bao nhiêu token; độ phân giải ảnh ↔ khả năng đọc chữ nhỏ; ảnh hưởng của tokenizer tiếng Việt | Paper Qwen2.5-VL, [Vintern-1B](https://arxiv.org/html/2408.12480v1) | Giảm dần `max_pixels`, vẽ đường cong độ chính xác |
| **Text embedding (E4)** | Bi-encoder, dense vs sparse vs multi-vector, giới hạn độ dài, chunking | [BGE-M3](https://huggingface.co/nrl-ai/vietnamese-embedding-bge-m3) | So chunk theo trang vs theo đoạn |
| **Visual retriever (E5)** | ColPali = VLM → vector/patch → MaxSim; DSE = 1 vector; tại sao late-interaction giữ chi tiết cục bộ nhưng tốn index; vector query nhìn vào patch nào | [ColPali](https://arxiv.org/html/2407.01449v5), [DSE](https://arxiv.org/html/2406.11251v2) | Vẽ heatmap similarity token↔patch cho vài câu hỏi |
| **Cấp vùng (E5)** | Cắt vùng bằng layout detector hay dùng patch của ColPali? Chọn đơn vị retrieval ảnh hưởng thế nào | [MMDocIR](https://arxiv.org/html/2501.08828v3) | So retrieval cấp trang vs cấp vùng cùng retriever |
| **LLM-judge** | Thiên lệch (độ dài, tự ưu ái), nên có đáp án chuẩn do người viết | [Judge's Verdict](https://arxiv.org/html/2510.09738v1), [Judging the Judges](https://arxiv.org/html/2406.12624v2) | Đo Cohen's κ judge vs bạn trên 50 câu |
| **LoRA/QLoRA (khóa luận)** | Rank, alpha, lớp nào được gắn adapter, quantization NF4, bộ nhớ | Paper LoRA, QLoRA | Chạy thử trên model 1B |
| **Agent (khóa luận)** | ReAct, Reflexion; state/loop trong LangGraph; điều kiện dừng | [Agentic RAG survey](https://arxiv.org/pdf/2501.09136), [ViDoRAG](https://arxiv.org/html/2502.18017v2) | Dựng agent 3 bước tối thiểu trước |

Gợi ý thứ tự đọc (2 tuần): ColPali → VisRAG → MMDocIR → survey Multimodal RAG → Vintern/ColVintern → ViDoRe V3 (phần phương pháp gán nhãn).

---

## 6. Đánh giá dataset thế nào

### 6.1 Đánh giá chất lượng dataset

| Khía cạnh | Cách làm |
|---|---|
| **Độ đồng thuận người gán** | 2 người gán độc lập trên ≥20–30% test → báo cáo **Cohen's κ** (không chỉ % đồng ý, xem [Judge's Verdict](https://arxiv.org/html/2510.09738v1)); với bbox dùng IoU |
| **Câu hỏi thật sự cần tài liệu** | Chạy LLM text-only không có tài liệu: nếu vẫn đúng nhiều → câu hỏi lộ đáp án/kiến thức chung; loại hoặc báo cáo tỷ lệ |
| **Câu hỏi thật sự cần ảnh** (với nhóm bảng/hình) | Kiểm tra trên OCR text: nếu OCR đủ trả lời thì câu đó không đo được giá trị của VLM |
| **Tính độc lập của câu hỏi** | Câu hỏi phải hiểu được khi không biết trang nào ("theo bảng bên dưới" là câu hỏi xấu cho retrieval trên kho) |
| **Đúng evidence** | Đáp án phải suy ra được **chỉ** từ vùng/trang evidence đã gán; kiểm mẫu ngẫu nhiên |
| **Rò rỉ dữ liệu** | Chia theo document (đã có); thêm: kiểm tra tài liệu test không nằm trong dữ liệu huấn luyện công khai của VLM (khó, ghi rõ giới hạn) |
| **Phân bố** | Báo cáo số câu theo loại (text/bảng/hình), số trang/tài liệu, độ dài câu trả lời, mức khó |

### 6.2 Cỡ mẫu: điểm yếu của scope hiện tại

Scope đặt 50–150 câu. Với n=100 và độ chính xác ~60%, khoảng tin cậy 95% xấp xỉ ±10 điểm % (≈ 1,96·√(0,6·0,4/100)). Hai hệ thống chênh 5 điểm **không phân biệt được**. Hệ quả:

- Khi chia thêm theo loại câu hỏi (bảng/hình), mỗi nhóm chỉ ~30–50 câu → kết luận rất yếu.
- Khuyến nghị: tăng test lên **≥300 câu** (dùng quy trình 3 bậc ở mục 2.2 để bù công), và báo cáo **bootstrap CI + kiểm định ghép cặp** (paired bootstrap hoặc McNemar) giữa các cặp E1–E5, thay vì chỉ bảng trung bình.
- Nếu không kịp: giữ n nhỏ nhưng **nói thẳng** đây là kết quả khảo sát (exploratory), không kết luận RQ mạnh.

### 6.3 Đánh giá bằng metric

- Câu trả lời: EM/F1 (chuẩn hóa dấu/hoa thường/Unicode NFC trước khi so), LLM-judge (có đáp án chuẩn do người viết; đo κ với người).
- Retrieval: Recall@K, MRR/nDCG; tách cấp trang và cấp vùng (IoU với bbox chuẩn).
- Hallucination: tách riêng "đúng nhưng không dựa vào evidence" khỏi "sai" để E2 vs E4/E5 có ý nghĩa.
- **Ablation tách lỗi**: chạy E5 với *evidence chuẩn (oracle retrieval)* để biết trần của VLM; nếu oracle vẫn thấp thì nút cổ chai là VLM chứ không phải retrieval.

---

## 7. Đề xuất việc làm tiếp theo

1. Đọc 5 paper ưu tiên (mục 5) và thêm bảng so sánh "giải pháp có sẵn vs ĐA2" vào chương Tổng quan.
2. Thử nhanh ColVintern-1B và BGE-M3 trên ~20 trang tài liệu tiếng Việt để xem E5 khả thi không (rủi ro lớn nhất đã ghi ở mục 12 của scope).
3. Đo VRAM/thời gian thực tế của Vintern-1B và Qwen2.5-VL 4-bit trên T4 → điền bảng mục 4 bằng số của bạn.
4. Chốt quy trình annotate 3 bậc và pilot 30 câu, đo κ.
5. Cập nhật scope: tăng test lên ≥300 (hoặc nêu rõ giới hạn), thêm oracle-retrieval ablation, thêm CI/kiểm định.

**Câu hỏi cần hỏi lại GVHD:**
- Ý "tài liệu tanh → translate" có đúng là *tiếng Anh → tiếng Việt* và nhằm làm dữ liệu fine-tune không?
- GVHD ưu tiên đóng góp là dataset/benchmark hay phương pháp (retriever/agent)?
- Mức chấp nhận về cỡ test set và việc dùng API trả phí cho sinh/chấm dữ liệu?
