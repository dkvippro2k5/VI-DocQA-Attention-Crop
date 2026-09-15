# ĐỀ XUẤT SCOPE ĐỒ ÁN 2

**Tên đề tài đề xuất:** Xây dựng và đánh giá pipeline baseline cho hỏi đáp trên tài liệu tiếng Việt: từ OCR+LLM đến VLM kết hợp Retrieval

**Vai trò:** Giai đoạn 1 của đề tài khóa luận tốt nghiệp *"Nghiên cứu Multimodal RAG kết hợp Vision-Language Model cho bài toán hỏi đáp trên tài liệu tiếng Việt"*. Phần Multimodal RAG (đa modality) và LoRA/QLoRA fine-tuning để dành cho khóa luận.

**Thời lượng:** 1 học kỳ (~14 tuần)

**Tài nguyên:** GPU T4 free (Google Colab / Kaggle)

---

## 1. Bối cảnh

Kế thừa từ đề xuất khóa luận gốc: tài liệu tiếng Việt thực tế (PDF, ảnh scan) chứa văn bản, bảng, hình ảnh và bố cục. VLM có thể xử lý trực tiếp nhưng gặp vấn đề khi số trang tăng (tốn tài nguyên, dễ hallucinate). RAG giúp truy xuất bằng chứng liên quan trước khi sinh câu trả lời. Đồ án 2 tập trung xây dựng và kiểm chứng các baseline trước khi mở rộng sang Multimodal RAG và fine-tuning ở khóa luận.

## 2. Phát biểu bài toán

Cho tập tài liệu tiếng Việt D và câu hỏi q, hệ thống cần xác định thành phần liên quan và sinh câu trả lời a dựa trên bằng chứng truy xuất được. Đồ án 2 giới hạn bằng chứng ở cấp độ **văn bản + trang** (chưa mở rộng sang vùng ảnh/bảng riêng biệt — dành cho khóa luận).

## 3. Mục tiêu

**Mục tiêu tổng quát:** Xây dựng pipeline baseline thống nhất (OCR+LLM, VLM trực tiếp, VLM+RAG) cho hỏi đáp tài liệu tiếng Việt và đánh giá định lượng vai trò của OCR, VLM và retrieval.

**Mục tiêu cụ thể:**
- Xây dựng tập dữ liệu nhỏ gồm tài liệu tiếng Việt và câu hỏi–câu trả lời có bằng chứng cấp trang.
- Triển khai baseline OCR + LLM.
- Triển khai VLM trực tiếp trên ảnh/trang tài liệu.
- Đánh giá ảnh hưởng của prompt engineering ở mức kiểm soát được.
- Xây dựng pipeline VLM + RAG (semantic retrieval text-only).
- So sánh các phương pháp bằng bộ tiêu chí thống nhất.
- Xây dựng hạ tầng code (data loader, embedding, vector store, eval script) tái sử dụng được cho khóa luận.

## 4. Câu hỏi nghiên cứu

- **RQ1:** VLM có cải thiện khả năng hỏi đáp trên tài liệu tiếng Việt so với pipeline OCR + LLM không?
- **RQ2:** Việc bổ sung retrieval (semantic search) có cải thiện kết quả khi tài liệu gồm nhiều trang không?

*(RQ3–RQ5 của đề tài gốc — về Multimodal RAG, LoRA và tổ hợp — để dành cho khóa luận.)*

## 5. Phạm vi và đối tượng nghiên cứu

- Đối tượng: VLM pretrained kích thước nhỏ/vừa (ưu tiên qua API hoặc quantized để chạy được trên T4 16GB free), kết hợp semantic retrieval trên kho tài liệu text.
- Phạm vi tài liệu: 1–2 loại tài liệu tiếng Việt (ví dụ: đề thi hoặc văn bản hành chính) để giới hạn công sức annotation.
- Không huấn luyện foundation model. Không thực hiện fine-tuning (LoRA/QLoRA để dành khóa luận).
- Không mở rộng retrieval sang đa modality (ảnh/bảng riêng biệt) — chỉ text + evidence cấp trang.

## 6. Các thí nghiệm

| TN | Phương pháp | Vai trò | Mức ưu tiên |
|---|---|---|---|
| E1 | OCR + LLM | Baseline văn bản | Bắt buộc |
| E2 | VLM trực tiếp | Baseline đa phương thức | Bắt buộc |
| E3 | VLM + Prompt Engineering | Kiểm tra ảnh hưởng prompt | Nên làm |
| E4 | VLM + RAG (semantic retrieval) | Baseline retrieval — trọng tâm ĐA2 | Bắt buộc |

*(E5 Multimodal RAG, E6 LoRA/QLoRA, E7 Multimodal RAG + LoRA — chuyển sang khóa luận. Nếu còn thời gian, có thể thử một bản E5 rất tối giản dạng proof-of-concept.)*

## 7. Dataset

- Quy mô: 50–150 cặp câu hỏi–câu trả lời.
- Đơn vị tổ chức: document, page, question, answer, evidence (cấp trang).
- Trường dữ liệu: `document_id`, `page_id`, `image`, `question`, `answer`, `evidence`, `question_type` (phân loại thô: text vs. bảng/hình).
- Chia tập train/validation/test theo **document** (không chia ngẫu nhiên theo câu hỏi) để tránh rò rỉ dữ liệu.
- Bộ dữ liệu và cách gán nhãn được thiết kế để **mở rộng trực tiếp** ở khóa luận (thêm evidence cấp vùng/bảng/hình, thêm loại câu hỏi multi-hop).

## 8. Mô hình và công nghệ

- **VLM:** ưu tiên mô hình nhỏ dạng API (vd. Gemini/GPT-4V-mini) hoặc VLM open-source quantized 4-bit chạy được trên T4 16GB free (Colab/Kaggle).
- **Retrieval:** embedding tiếng Việt/multilingual (sentence-transformers) + vector store nhẹ (FAISS/Chroma), semantic search đơn (không hybrid/rerank).
- **Prototype (nếu còn thời gian):** giao diện web nhẹ để tải tài liệu, nhập câu hỏi, xem câu trả lời và trang bằng chứng.

## 9. Phương pháp đánh giá

| Nhóm | Chỉ số | Giữ trong ĐA2 | Để khóa luận |
|---|---|---|---|
| Câu trả lời | Exact Match, Token-level F1, **LLM-judge semantic similarity** | ✅ | Human evaluation quy mô lớn |
| Retrieval | **Recall@K (semantic/embedding search)** | ✅ | Hybrid lexical+semantic fusion, reranker |
| Độ tin cậy | **Evidence accuracy / gắn nhãn hallucination thủ công** trên tập test nhỏ (câu trả lời có được hỗ trợ bởi bằng chứng truy xuất không) | ✅ | Đo GPU/RAM chi tiết, groundedness score tự động |

**Lý do giữ LLM-judge:** EM/F1 đánh giá sai thấp với câu trả lời tiếng Việt diễn đạt tự do đúng nghĩa nhưng khác từ ngữ; chi phí thêm không đáng kể vì chỉ cần 1 API call/câu trên tập nhỏ.

**Lý do giữ đo hallucination (evidence accuracy):** đây là luận điểm cốt lõi cần chứng minh — RAG (E4) có giảm hallucination so với VLM trực tiếp (E2) không. Không đo thì bảng so sánh E1–E4 mất phần thuyết phục nhất.

**Lý do hoãn hybrid + reranker:** đây là bước tối ưu retrieval thêm, không cần thiết để trả lời RQ2 (semantic search đơn đã đủ so sánh có/không retrieval). Với dataset nhỏ, lexical (BM25) khó thể hiện lợi thế rõ rệt so với semantic — độ phân giải thống kê thấp. Phù hợp hơn khi mở rộng ở khóa luận với dataset lớn và nhiều loại bằng chứng hơn.

## 10. Đóng góp dự kiến

1. Dataset nhỏ (50–150 Q&A) có evidence cấp trang — tái sử dụng và mở rộng ở khóa luận.
2. Pipeline baseline thống nhất OCR+LLM / VLM / VLM+RAG, code tái sử dụng được.
3. Bảng so sánh định lượng E1–E4, trả lời RQ1 và RQ2 bằng thực nghiệm.
4. Hạ tầng code (data loader, embedding, vector store, eval script với EM/F1/LLM-judge/Recall@K/evidence accuracy) làm nền cho khóa luận bổ sung Multimodal RAG và LoRA.

## 11. Kế hoạch thực hiện (14 tuần)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 1–3 | Khảo sát tài liệu liên quan, chốt phạm vi, chọn model/API | Tổng quan + đặc tả bài toán |
| 4–6 | Thu thập & gán nhãn dataset nhỏ | Dataset v1 |
| 7–8 | Xây dựng E1 (OCR+LLM), E2 (VLM trực tiếp) | Baseline results |
| 9–10 | Xây dựng E3 (Prompt engineering), E4 (VLM+RAG) | Kết quả retrieval baseline |
| 11–13 | So sánh, phân tích lỗi, viết báo cáo | Bảng thí nghiệm + báo cáo |
| 14 | Chuẩn bị bảo vệ | Đồ án hoàn chỉnh |

## 12. Rủi ro và phương án dự phòng

| Rủi ro | Ảnh hưởng | Phương án |
|---|---|---|
| GPU T4 free không đủ / hết quota | Không chạy được VLM | Ưu tiên VLM qua API, quantization, giới hạn batch |
| Dataset annotation tốn thời gian | Chậm tiến độ | Giảm quy mô xuống 50 câu, ưu tiên chất lượng test set |
| OCR tiếng Việt sai nhiều | Baseline E1 thấp | Dùng VLM trực tiếp (E2) làm đối chứng, không phụ thuộc hoàn toàn vào E1 |
| LLM-judge không ổn định/tốn API quota | Đánh giá thiếu tin cậy | Kết hợp với EM/F1, kiểm tra chéo thủ công trên mẫu nhỏ |

## 13. Kết quả đầu ra dự kiến

- Dataset nhỏ hỏi đáp tài liệu tiếng Việt (evidence cấp trang), sẵn sàng mở rộng.
- Mã nguồn pipeline OCR+LLM, VLM, VLM+RAG và script đánh giá.
- Bảng kết quả so sánh E1–E4 (EM, F1, LLM-judge, Recall@K, evidence accuracy).
- Phân tích lỗi sơ bộ theo loại câu hỏi.
- Báo cáo đồ án 2, làm nền cho chương "Baseline" của khóa luận tốt nghiệp.

## 14. Liên kết với khóa luận tốt nghiệp

Đồ án 2 chỉ thực hiện E1–E4 và hạ tầng liên quan. Các phần sau được chuyển nguyên sang khóa luận:

- **E5 – Multimodal RAG:** mở rộng retrieval sang ảnh trang/vùng/bảng, contribution chính của khóa luận.
- **E6 – VLM + LoRA/QLoRA:** fine-tuning nhẹ thích nghi tài liệu tiếng Việt.
- **E7 – Multimodal RAG + LoRA:** mô hình cuối, kết hợp cả hai hướng.
- **RQ3–RQ5:** các câu hỏi nghiên cứu về multimodal retrieval và fine-tuning.
- Đánh giá nâng cao: hybrid lexical+semantic fusion, reranker, human evaluation quy mô lớn, đo tài nguyên hệ thống chi tiết.
- Mở rộng dataset: evidence cấp vùng/bảng/hình, câu hỏi multi-hop.
