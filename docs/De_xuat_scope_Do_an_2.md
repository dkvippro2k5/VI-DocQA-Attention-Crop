# ĐỀ XUẤT SCOPE ĐỒ ÁN 2

**Tên đề tài đề xuất:** Can thiệp training-free dựa trên attention cho Vision-Language Model trong hỏi đáp tài liệu tiếng Việt: so sánh với OCR+LLM và các pipeline Multimodal RAG

**Vai trò:** Giai đoạn 1 của khóa luận tốt nghiệp. ĐA2 không huấn luyện mô hình; phần fine-tuning (LoRA nhắm vào layer/head đã chẩn đoán) và agent để dành cho khóa luận.

**Thời lượng:** 1 học kỳ (~14 tuần)

**Tài nguyên:** GPU T4 16GB free (Google Colab / Kaggle), API cho LLM-judge và sinh nháp dữ liệu

**Tài liệu liên quan:** [Related_work_References.md](Related_work_References.md), [Nghien_cuu_theo_gop_y_GVHD.md](Nghien_cuu_theo_gop_y_GVHD.md)

---

## 1. Bối cảnh và khoảng trống nghiên cứu

Tài liệu tiếng Việt thực tế (PDF, ảnh scan) chứa văn bản, bảng, hình và bố cục. Có hai cách tiếp cận chính:

- **OCR + LLM:** OCR tiếng Việt hay sai dấu thanh, và lỗi OCR lan sang toàn bộ pipeline RAG (OHRBench – *OCR Hinders RAG*, ICCV 2025).
- **VLM đọc trực tiếp ảnh trang, kết hợp truy xuất bằng ảnh** (ColPali, VisRAG – ICLR 2025): vượt RAG trên OCR text ở tiếng Anh. Tuy vậy VLM gặp khó với chi tiết nhỏ, và với ngôn ngữ ít tài nguyên có thể "đoán" thay vì "đọc" (*Reading or Guessing?*, 2026).

Một hướng sửa không cần huấn luyện là **crop theo attention**: dùng attention của chính VLM để xác định vùng cần nhìn, phóng to vùng đó rồi hỏi lại.
- **ViCrop** (*MLLMs Know Where to Look*, ICLR 2025) cải thiện TextVQA, DocVQA mà không cần train.
- **HAVC** (2026) chọn ra các attention head "biết đọc chữ" bằng một tác vụ chẩn đoán OCR, rồi chỉ dùng các head đó để crop, và vượt ViCrop.

**Khoảng trống (theo khảo sát hiện tại):**
1. Các phương pháp crop training-free mới được đánh giá trên tiếng Anh, ảnh đơn, chưa đặt trong pipeline RAG tài liệu nhiều trang.
2. HAVC chẩn đoán head bằng chữ Latin không dấu (Synthdog). Chưa ai kiểm tra liệu head chọn theo chữ tiếng Anh có định vị tốt chữ **tiếng Việt có dấu** hay không.
3. Chưa có so sánh có kiểm soát giữa OCR+LLM, VLM+RAG text và Multimodal RAG trên tài liệu tiếng Việt.

## 2. Phát biểu bài toán

Cho kho tài liệu tiếng Việt D (nhiều trang) và câu hỏi q, hệ thống truy xuất trang liên quan rồi sinh câu trả lời a dựa trên bằng chứng. ĐA2 nghiên cứu: với cùng retriever và cùng VLM, **can thiệp training-free vào cách VLM nhìn ảnh** (crop theo attention của các head được chọn) có cải thiện câu trả lời không, và cách chọn head theo ngôn ngữ đích có quan trọng không.

## 3. Mục tiêu

**Mục tiêu tổng quát:** Đánh giá có kiểm soát các pipeline hỏi đáp tài liệu tiếng Việt (OCR+LLM, VLM+RAG text, Multimodal RAG) và đề xuất, kiểm chứng một biến thể can thiệp crop training-free với bước chẩn đoán head bằng chữ tiếng Việt có dấu.

**Mục tiêu cụ thể:**
- Xây dựng dataset hỏi đáp tài liệu tiếng Việt có tập dev và test chia theo tài liệu, có bbox vùng đáp án và nhãn loại câu hỏi.
- Triển khai nhánh A, B, C1, C2 (mục 6) trên cùng dataset và cùng giao thức đánh giá.
- Chẩn đoán head/layer của VLM: head nào định vị đúng vùng chữ tiếng Việt, so với head chọn theo chữ tiếng Anh.
- Triển khai ViCrop (C3), HAVC tái hiện (C4) và HAVC với chẩn đoán tiếng Việt (C5).
- Phân tích kết quả theo loại câu hỏi (chi tiết nhỏ / dấu thanh / bảng / văn bản thường), có kiểm định thống kê.

## 4. Câu hỏi nghiên cứu

- **RQ1:** Trên tài liệu tiếng Việt, nhánh nào trong OCR+LLM, VLM+RAG text và Multimodal RAG cho kết quả tốt nhất, và khác biệt nằm ở loại câu hỏi nào?
- **RQ2:** Can thiệp crop training-free (ViCrop, HAVC) có cải thiện VLM trên tài liệu tiếng Việt không, đặc biệt ở câu hỏi cần đọc chi tiết nhỏ / dấu thanh?
- **RQ3 (đóng góp chính):** Chọn head bằng tác vụ chẩn đoán **tiếng Việt có dấu** có cho crop tốt hơn chọn head bằng chữ tiếng Anh như HAVC gốc không?

**Giả thuyết cho RQ3:** dấu thanh là chi tiết nhỏ, nằm trên/dưới ký tự; head định vị tốt chữ Latin không dấu chưa chắc khoanh đủ vùng dấu. Kết quả âm (không khác biệt) vẫn được báo cáo như một phát hiện.

## 5. Phạm vi

- **Không huấn luyện / fine-tune** bất kỳ mô hình nào ở ĐA2. Mọi can thiệp chỉ thay đổi cách mô hình chạy lúc inference; trọng số giữ nguyên, nên trường hợp xấu nhất là quay về baseline.
- VLM open-source cỡ 1–3B (chạy được và lấy được attention trên T4). Cỡ 7B 4-bit chỉ dùng nếu đủ VRAM.
- 1–2 loại tài liệu tiếng Việt (ví dụ văn bản hành chính, đề thi).
- Retrieval single-shot, không hybrid/reranker, không agent.
- **Không đưa vào ĐA2:** text–image fusion (UniDoc-Bench), PAI, gộp patch retriever (Light-ColPali), prompt engineering. Chỉ trích trong Related Work.

## 6. Các nhánh thí nghiệm

Tất cả nhánh chạy trên **cùng tập test, cùng câu hỏi, cùng prompt sinh đáp án**.

| Nhánh | Pipeline | Vai trò |
|---|---|---|
| **A** | OCR + retrieval text + LLM | Baseline văn bản |
| **B** | Retrieval text (trên OCR) + VLM đọc ảnh trang truy xuất được | Baseline VLM + RAG text |
| **C1** | ColVintern-1B (retriever ảnh) + Vintern-1B | Multimodal RAG có sẵn #1 |
| **C2** | ColQwen (retriever ảnh) + Qwen2.5-VL-3B | Multimodal RAG có sẵn #2 |
| **C3** | Nhánh C tốt nhất + **ViCrop** (`rel_att`, theo code gốc) | Can thiệp đã công bố |
| **C4** | Nhánh C tốt nhất + **HAVC** (head chọn bằng chữ tiếng Anh, theo paper) | Can thiệp đã công bố (tái hiện) |
| **C5** | Nhánh C tốt nhất + **HAVC với chẩn đoán tiếng Việt có dấu** | **Đề xuất của ĐA2** |

- "Nhánh C tốt nhất" được chọn trên **tập dev**, không phải test.
- **So sánh chính (khai báo trước):** C5 vs C4. So sánh phụ: C3/C4 vs nhánh C gốc; A vs B vs C1 vs C2.
- Ablation phụ: chạy C3–C5 với **trang đúng (oracle retrieval)** để tách lỗi retrieval khỏi lỗi đọc của VLM.

**Chẩn đoán head (dùng cho C4, C5):** theo quy trình HAVC — đưa ảnh chữ vào VLM, khi model đọc đúng thì ghi vị trí đỉnh attention của từng head, giữ head có điểm trùng vùng chữ ≥ ngưỡng, rồi lọc bằng entropy không gian và độ nhạy gradient (K = 8, ngưỡng 0,5, α = 0,4, τ = 0,1 theo paper; tinh chỉnh trên dev).
- C4: ảnh chữ tiếng Anh (Synthdog-en, như paper).
- C5: ảnh chữ tiếng Việt có dấu (render từ văn bản tiếng Việt, cùng số lượng và cùng font/cỡ chữ với C4) để **chỉ thay đổi một biến là ngôn ngữ**.

## 7. Dataset

| Tập | Cỡ | Dùng cho |
|---|---|---|
| **Dev** | 100–150 câu | Chọn nhánh C tốt nhất, chẩn đoán/tinh chỉnh siêu tham số crop |
| **Test** | Tối thiểu 300, mục tiêu 400–500 câu | Chạy **một lần** với cấu hình đã chốt trên dev |
| Train | Không cần ở ĐA2 | Dành cho LoRA ở khóa luận |
| Ảnh chẩn đoán head | ~1.000 ảnh tiếng Anh + ~1.000 ảnh tiếng Việt (tự sinh) | C4, C5 |

- **Chia theo tài liệu:** không tài liệu nào xuất hiện ở cả dev và test. Lập sẵn danh sách tài liệu dev/test; các tài liệu này không bao giờ dùng để sinh dữ liệu train ở khóa luận.
- **Chia tầng test:** ≥150 câu thuộc nhóm "chi tiết nhỏ / dấu thanh / ô bảng", phần còn lại là câu văn bản thường để đối chứng.
- **Trường dữ liệu:** `document_id`, `page_id`, `image`, `question`, `answer`, `evidence_bbox`, `question_type` (small_detail / diacritic / table / text), `split`.
- **Quy trình tạo:** VLM mạnh sinh nháp câu hỏi + đáp án + bbox → lọc tự động (bỏ câu trả lời được khi không có tài liệu, câu mơ hồ, câu không tự đủ nghĩa, trùng lặp) → người duyệt 100% dev và test.
- **Kiểm tra chất lượng:** 2 người gán độc lập trên ≥20% test, báo cáo Cohen's κ (đáp án) và IoU (bbox).
- **Link dataset:** https://huggingface.co/datasets/trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning

## 8. Mô hình và công nghệ

| Thành phần | Lựa chọn |
|---|---|
| OCR (A, B) | OCR tiếng Việt open-source (chốt ở tuần 1–3, đo CER trên mẫu nhỏ) |
| Embedding text (A, B) | BGE-M3 hoặc bản fine-tune tiếng Việt; FAISS |
| LLM (A) | LLM cùng cỡ với VLM, hoặc API |
| Retriever ảnh | ColVintern-1B (C1), ColQwen (C2) |
| VLM | Vintern-1B (C1), Qwen2.5-VL-3B (C2) |
| Can thiệp | Code ViCrop ([mllms_know](https://github.com/saccharomycetes/mllms_know), MIT, đã hỗ trợ Qwen2.5-VL); HAVC **tự cài đặt** trên nền code ViCrop (paper không công bố code) |
| Đánh giá | LLM-judge qua API |

Lưu ý kỹ thuật:
- Lấy attention cần chế độ eager attention; T4 không hỗ trợ bf16 gốc và FlashAttention-2 → dùng fp16, giới hạn `max_pixels`.
- HAVC và `grad_att` cần backward lúc inference → tốn VRAM; đo trước trên dev.
- Vintern chưa được code ViCrop hỗ trợ → phải viết lớp tương tự `qwen2_5_methods.py`.

## 9. Đánh giá

| Nhóm | Chỉ số |
|---|---|
| Câu trả lời | EM, token-level F1 (chuẩn hóa Unicode NFC, dấu, hoa/thường), LLM-judge có đáp án chuẩn (đo Cohen's κ với người trên ~50 câu) |
| Retrieval | Recall@K, MRR cấp trang |
| Định vị của can thiệp | Tỷ lệ vùng crop chứa bbox đáp án; IoU giữa vùng crop và bbox |
| Chẩn đoán head | Độ trùng attention với vùng chữ (head chọn bằng tiếng Anh vs tiếng Việt), riêng cho vùng dấu thanh |
| Chi phí | Thời gian/câu, VRAM đỉnh, số forward/backward pass |

**Kiểm định thống kê:**
- So sánh ghép cặp giữa các nhánh trên cùng câu hỏi bằng **McNemar** (đúng/sai) và **paired bootstrap** (F1, LLM-judge), báo cáo khoảng tin cậy 95%.
- So sánh chính (C5 vs C4) khai báo trước; các so sánh phụ hiệu chỉnh **Holm**.
- Ước lượng sơ bộ: với ~15–20% câu đổi kết quả giữa hai cấu hình, 300 câu phát hiện được chênh lệch khoảng 6–7 điểm %; 5 điểm % cần ~470 câu. Tính lại sau pilot.

## 10. Đóng góp dự kiến

1. **Phương pháp:** biến thể HAVC với tác vụ chẩn đoán head theo ngôn ngữ đích (tiếng Việt có dấu), kèm so sánh có kiểm soát với ViCrop và HAVC gốc (RQ2, RQ3).
2. **Phân tích:** head/layer nào của VLM định vị chữ tiếng Việt, và khác gì so với chữ tiếng Anh.
3. **Đánh giá:** so sánh có kiểm soát OCR+LLM, VLM+RAG text và hai Multimodal RAG trên tài liệu tiếng Việt, phân tích theo loại câu hỏi (RQ1).
4. **Tài nguyên:** dataset dev/test có bbox và nhãn loại câu hỏi; bộ ảnh chẩn đoán tiếng Việt; code tái hiện HAVC.

## 11. Kế hoạch thực hiện (14 tuần)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 1–2 | Đọc toàn văn bài cốt lõi (ViCrop, HAVC, OHRBench, Reading or Guessing, ColPali, VisRAG); chốt OCR và model | Related Work bản nháp |
| 2–3 | **Pilot:** chạy ViCrop trên Qwen2.5-VL-3B với ~20 trang tiếng Việt; đo VRAM, thời gian | Quyết định khả thi (go/no-go) |
| 3–6 | Thu thập tài liệu, sinh nháp + duyệt dev và test; đo κ | Dataset v1 |
| 5–7 | Nhánh A, B, C1, C2 + script đánh giá | Kết quả RQ1 trên dev |
| 7–9 | **Tự cài HAVC** trên nền code ViCrop; sinh bộ ảnh chẩn đoán tiếng Anh và tiếng Việt; chẩn đoán head | C4, C5 chạy được trên dev |
| 9–10 | Port ViCrop sang Vintern (nếu nhánh C tốt nhất là C1); tinh chỉnh siêu tham số trên dev; chốt cấu hình | Cấu hình cuối |
| 11 | Chạy **một lần** tất cả nhánh trên test; oracle-retrieval ablation | Bảng kết quả |
| 12 | Kiểm định thống kê, phân tích lỗi theo loại câu hỏi, phân tích head | Phân tích |
| 13 | Viết báo cáo | Báo cáo |
| 14 | Chuẩn bị bảo vệ | Đồ án hoàn chỉnh |

## 12. Rủi ro và phương án dự phòng

| Rủi ro | Ảnh hưởng | Phương án |
|---|---|---|
| Lấy attention / backward vượt VRAM T4 | Không chạy được C3–C5 | Dùng ViCrop `rel_att` (chỉ forward); giảm `max_pixels`; chuyển sang FOCUS (không cần gradient); dùng Kaggle 2×T4 |
| Tự cài HAVC sai khác paper | C4 không tái hiện được | Kiểm tra trên TextVQA một tập nhỏ, so với số trong paper; báo cáo rõ sai khác |
| Can thiệp không cải thiện trên tiếng Việt | RQ2/RQ3 cho kết quả âm | Vẫn báo cáo được; bổ sung phân tích chẩn đoán head để giải thích nguyên nhân |
| VLM 1–3B định vị chữ tiếng Việt kém ở mọi head | C5 không có cơ sở | Phát hiện ở pilot tuần 2–3; nếu vậy chuyển trọng tâm sang phân tích (RQ1 + chẩn đoán) |
| Annotation chậm | Không đủ 300 câu test | Ưu tiên test hơn dev; giữ đủ nhóm chi tiết nhỏ; nếu dưới 300 câu thì báo cáo là kết quả khảo sát, không kết luận mạnh |
| LLM-judge thiếu ổn định | Đánh giá kém tin cậy | Đáp án chuẩn do người viết; đo κ với người; báo cáo kèm EM/F1 |

## 13. Kết quả đầu ra

- Dataset dev/test tài liệu tiếng Việt có bbox và nhãn loại câu hỏi; bộ ảnh chẩn đoán head tiếng Việt.
- Code các nhánh A, B, C1–C5, code tái hiện HAVC, script đánh giá + kiểm định thống kê.
- Bảng kết quả, phân tích theo loại câu hỏi, phân tích head/layer, case study crop đúng/sai.
- Báo cáo ĐA2.

## 14. Liên kết với khóa luận

ĐA2 không huấn luyện. Khóa luận mở rộng trên kết quả chẩn đoán của ĐA2:

- **LoRA nhắm vào layer/head đã chẩn đoán** so với LoRA toàn bộ và so với can thiệp training-free C5. Nếu fine-tune làm kết quả tệ đi, kết quả training-free của ĐA2 vẫn đứng vững.
- **Dữ liệu train** sinh synthetic theo cùng quy trình ở mục 7, chỉ từ tài liệu ngoài danh sách dev/test.
- **Agentic RAG (tùy chọn):** đưa crop theo attention thành một tool mà agent tự quyết định gọi, so với crop cố định ở C5.
