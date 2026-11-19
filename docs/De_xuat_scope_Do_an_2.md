# ĐỀ XUẤT SCOPE ĐỒ ÁN 2

**Tên đề tài đề xuất:** Can thiệp dựa trên attention cho Vision-Language Model khi đọc trang tài liệu tiếng Việt: crop training-free theo head chẩn đoán bằng chữ có dấu và LoRA nhắm vào các head đó

*Tên cũ (đến 14/11/2026): Can thiệp training-free dựa trên attention cho Vision-Language Model trong hỏi đáp tài liệu tiếng Việt: so sánh với OCR+LLM và các pipeline Multimodal RAG*

**Vai trò:** Giai đoạn 1 của khóa luận tốt nghiệp. ĐA2 tập trung vào bài toán hẹp và sâu: VLM đọc chi tiết nhỏ và dấu thanh trên **một trang**. Gồm can thiệp training-free (crop theo attention) và LoRA nhắm vào các head đã chẩn đoán. Truy xuất trên tài liệu nhiều trang (OCR+LLM, RAG văn bản, Multimodal RAG) và agent để dành cho khóa luận.

**Thời lượng:** 1 học kỳ (~14 tuần)

**Tài nguyên:** GPU T4 16GB free (Google Colab / Kaggle); LLM-judge chạy cục bộ trên Kaggle (`llama.cpp`); API cho sinh nháp dữ liệu

**Tài liệu liên quan:** [Related_work_References.md](Related_work_References.md), [Giai_dap_gop_y_GVHD_tuan_4.md](Giai_dap_gop_y_GVHD_tuan_4.md), [Nguon_du_lieu_tieng_Viet.md](Nguon_du_lieu_tieng_Viet.md)

**Lịch sử thay đổi:**
- 15/11/2026, theo góp ý của GVHD sau báo cáo tuần 4 (GVHD đã đồng ý 5 điểm ở mục 5 file giải đáp):
  - gác phần truy xuất sang khóa luận;
  - thêm thiết kế 2 × 2 (ngôn ngữ × loại ảnh);
  - thêm baseline theo bậc;
  - bỏ bước khôi phục dấu, chỉ chấm thêm điểm khi bỏ dấu;
  - đưa LoRA (C6, C7) vào ĐA2, train trên dữ liệu tiếng Việt công khai có người viết hoặc người duyệt.

---

## 1. Bối cảnh và khoảng trống nghiên cứu

Tài liệu tiếng Việt thực tế (PDF, ảnh scan) có chữ nhỏ, dày đặc và dấu thanh. VLM đọc trực tiếp ảnh trang đã vượt OCR+LLM ở tiếng Anh (ColPali, VisRAG – ICLR 2025). Tuy vậy VLM vẫn gặp khó với chi tiết nhỏ, và với ngôn ngữ ít tài nguyên có thể "đoán" thay vì "đọc" (*Reading or Guessing?*, 2026). Với tiếng Việt, một dấu sai là thành một từ khác.

Một hướng sửa không cần huấn luyện là **crop theo attention**: dùng attention của chính VLM để xác định vùng cần nhìn, phóng to vùng đó rồi hỏi lại.
- **ViCrop** (*MLLMs Know Where to Look*, ICLR 2025) cải thiện TextVQA, DocVQA mà không cần train.
- **HAVC** (2026) chọn ra các attention head "biết đọc chữ" bằng một tác vụ chẩn đoán OCR, rồi chỉ dùng các head đó để crop, và vượt ViCrop.

**Khoảng trống (theo khảo sát hiện tại):**
1. ViCrop và HAVC được đánh giá chủ yếu trên ảnh cảnh có ít chữ, bằng tiếng Anh, với model độ phân giải thấp (LLaVA-1.5, InstructBLIP). HAVC không có bộ tài liệu nào; ViCrop chỉ có DocVQA, mức tăng nhỏ. Chưa rõ các phương pháp này có còn tác dụng trên trang tài liệu dày chữ và trên tiếng Việt hay không.
2. HAVC chẩn đoán head bằng chữ Latin không dấu (Synthdog). Chưa ai kiểm tra liệu head chọn theo chữ tiếng Anh có định vị tốt chữ **tiếng Việt có dấu** hay không.
3. Chưa có so sánh giữa can thiệp training-free theo head và **fine-tune nhắm đúng các head đó** (LoRA), nhất là khi không có dữ liệu train cùng miền cho tiếng Việt.

## 2. Phát biểu bài toán

Cho một ảnh trang (hoặc ảnh có chữ) và câu hỏi q, VLM sinh câu trả lời a bằng cách đọc chữ trên ảnh. Trang đưa vào là trang chứa đáp án (chế độ "trang đúng"), không có bước truy xuất. ĐA2 nghiên cứu:
- **can thiệp training-free vào cách VLM nhìn ảnh** (crop theo attention của các head được chọn) có cải thiện câu trả lời không, và cách chọn head theo ngôn ngữ đích có quan trọng không;
- **fine-tune nhẹ nhắm vào chính các head đó** (LoRA) có hơn can thiệp training-free không.

## 3. Mục tiêu

**Mục tiêu tổng quát:** Đánh giá có kiểm soát các can thiệp dựa trên attention cho VLM khi đọc ảnh có chữ, theo hai yếu tố ngôn ngữ (Anh / Việt) và loại ảnh (ảnh cảnh / tài liệu dày chữ). Đề xuất và kiểm chứng (i) bước chẩn đoán head bằng chữ tiếng Việt có dấu cho crop training-free, (ii) LoRA nhắm vào các head đã chẩn đoán.

**Mục tiêu cụ thể:**
- Xây dựng dataset hỏi đáp tài liệu tiếng Việt có dev và test chia theo tài liệu, có bbox vùng đáp án và nhãn loại câu hỏi.
- Lấy mẫu ba bộ công khai cho thiết kế 2 × 2: TextVQA, DocVQA, ViTextVQA.
- Chạy baseline theo bậc: model gốc → tiền xử lý → can thiệp.
- Chẩn đoán head/layer của VLM: head nào định vị đúng vùng chữ tiếng Việt, so với head chọn theo chữ tiếng Anh (đã làm).
- Triển khai ViCrop (C3), HAVC tái hiện (C4), HAVC với chẩn đoán tiếng Việt (C5), crop ngẫu nhiên (đối chứng).
- Triển khai LoRA nhắm head (C6) và LoRA toàn bộ (C7).
- Phân tích kết quả theo loại câu hỏi (chi tiết nhỏ / dấu thanh / bảng / văn bản thường), có kiểm định thống kê.

## 4. Câu hỏi nghiên cứu

- **RQ1:** Mức cải thiện của crop theo attention (ViCrop, HAVC) so với model gốc thay đổi thế nào khi đổi **loại ảnh** (ảnh cảnh / tài liệu dày chữ) và **ngôn ngữ** (Anh / Việt)?
- **RQ2:** Trên tài liệu tiếng Việt, crop training-free có cải thiện VLM không, đặc biệt ở câu hỏi cần đọc chi tiết nhỏ / dấu thanh?
- **RQ3 (đóng góp chính):** Chọn head bằng tác vụ chẩn đoán **tiếng Việt có dấu** có cho crop tốt hơn chọn head bằng chữ tiếng Anh như HAVC gốc không?
- **RQ4:** LoRA chỉ gắn vào các head đã chẩn đoán, train trên dữ liệu VQA tiếng Việt giàu chữ nhưng không phải tài liệu, có chuyển được sang tài liệu tiếng Việt không? Nó có hơn LoRA toàn bộ và hơn crop training-free C5 không?

**Giả thuyết cho RQ3:** dấu thanh là chi tiết nhỏ, nằm trên/dưới ký tự; head định vị tốt chữ Latin không dấu chưa chắc khoanh đủ vùng dấu. Kết quả âm (không khác biệt) vẫn được báo cáo như một phát hiện. Kết quả chẩn đoán đã có: head chọn bằng tiếng Việt định vị chữ tiếng Việt tốt hơn head chọn bằng tiếng Anh khoảng 2 điểm %, rõ hơn ở chữ có dấu (báo cáo tuần 4). Câu hỏi còn mở là chênh lệch này có chuyển thành chênh lệch về câu trả lời không.

**Giả thuyết cho RQ4:** LoRA nhắm head ít tham số hơn nhiều so với LoRA toàn bộ nên ít bị quá khớp khi dữ liệu nhỏ và khác miền. Kết quả âm (không chuyển được sang tài liệu, hoặc thua C5) vẫn được báo cáo.

## 5. Phạm vi

- **Huấn luyện chỉ ở nhánh C6, C7**, dưới dạng LoRA/QLoRA; trọng số gốc giữ nguyên. Các nhánh còn lại là can thiệp lúc inference, trọng số giữ nguyên.
- **Dữ liệu train chỉ lấy từ bộ công khai có người viết hoặc người duyệt** (mục 7). Không dùng nhãn LLM chưa duyệt. Không dùng bất kỳ tài liệu nào của dev/test.
- VLM open-source cỡ 1–3B (chạy được và lấy được attention trên T4). Model chính là Qwen2.5-VL-3B; Vintern-1B chỉ dùng ở baseline. Cỡ 7B 4-bit chỉ dùng nếu đủ VRAM.
- Đầu vào là một ảnh trang (chế độ trang đúng).
- **Không đưa vào ĐA2:**
  - truy xuất trên tài liệu nhiều trang (OCR+LLM, RAG văn bản, ColVintern/ColQwen), chuyển sang khóa luận;
  - khôi phục dấu cho văn bản;
  - text–image fusion (UniDoc-Bench), PAI, gộp patch retriever (Light-ColPali), prompt engineering.
  
  Các hướng này chỉ trích trong Related Work.

## 6. Các nhánh thí nghiệm

Tất cả nhánh chạy trên **cùng tập test, cùng câu hỏi, cùng prompt sinh đáp án, cùng script chấm**.

| Nhánh | Nội dung | Vai trò |
|---|---|---|
| **B0** | Model gốc: Qwen2.5-VL-3B (và Vintern-1B để tham khảo), ảnh nguyên, `max_pixels` mặc định của dự án | Baseline bậc 0 |
| **B1** | Tiền xử lý: (a) tăng `max_pixels`, cân số token ảnh với các nhánh crop; (b) xử lý ảnh (phóng to, làm nét, xoay thẳng trang scan) | Baseline bậc 1: biện pháp đơn giản không đụng tới model |
| **C3** | Qwen2.5-VL-3B + **ViCrop** (`rel_att`, theo code gốc) | Can thiệp đã công bố |
| **C4** | Qwen2.5-VL-3B + **HAVC** (head chọn bằng chữ tiếng Anh, theo paper) | Can thiệp đã công bố (tái hiện) |
| **C5** | Qwen2.5-VL-3B + **HAVC với chẩn đoán tiếng Việt có dấu** | **Đề xuất của ĐA2** |
| **R** | Qwen2.5-VL-3B + crop ngẫu nhiên cùng kích thước với C5 | Đối chứng: lợi ích có đến từ vị trí crop không |
| **O** | Qwen2.5-VL-3B + crop theo `evidence_bbox` chuẩn, cùng kích thước và cách đưa ảnh với C5 (chỉ VisR-Bench tiếng Việt) | Trần trên của crop; chỉ để phân tích (mục 6.1) |
| **C6** | Qwen2.5-VL-3B + **LoRA nhắm head**: chỉ gắn vào phép chiếu attention của các layer chứa head đã chẩn đoán ở C5 (biến thể chặt: chặn cập nhật ngoài các head đã chọn) | Đề xuất của ĐA2 (fine-tune) |
| **C7** | Qwen2.5-VL-3B + **LoRA toàn bộ**: mọi layer attention, cùng dữ liệu và cùng số bước với C6 | Đối chứng cho C6 |

- **So sánh chính (khai báo trước):** C5 vs C4, trên tài liệu tiếng Việt.
- **So sánh phụ (hiệu chỉnh Holm):**
  - C3/C4/C5 vs B0 và vs B1;
  - C5 vs R;
  - C6 vs C7;
  - C6 vs C5;
  - mức chênh của từng can thiệp giữa 4 ô của thiết kế 2 × 2.
- Mọi siêu tham số (của crop và của LoRA) chọn trên **dev**, không chọn trên test.
- Mỗi nhánh chọn **một cấu hình chung cho cả 4 bộ** trên dev gộp (khoảng 1.150 câu), không chọn riêng từng bộ: 300 câu dev mỗi bộ vẫn quá ít để phân biệt các cấu hình chênh nhau vài điểm, và thiết kế 2 × 2 so mức chênh của *cùng một* can thiệp giữa các ô (chốt 18/11/2026).
- **Chỉ số chính** cho so sánh chính và cho quy tắc mở rộng (mục 6.1): tỉ lệ đúng theo LLM-judge có đáp án chuẩn, trên nhóm `short` của VisR-Bench tiếng Việt; EM, F1, tỉ lệ chứa đáp án báo cáo kèm (sửa đổi 18/11/2026, lý do ở mục 6.1).

**Chẩn đoán head (dùng cho C4, C5, C6):** theo quy trình HAVC. Đưa ảnh chữ vào VLM; khi model đọc đúng thì ghi vị trí đỉnh attention của từng head; giữ head có điểm trùng vùng chữ ≥ ngưỡng; rồi lọc bằng entropy không gian và độ nhạy gradient. Siêu tham số theo paper: K = 8, ngưỡng 0,5, α = 0,4, τ = 0,1; tinh chỉnh trên dev.
- C4: ảnh chữ tiếng Anh (như paper).
- C5: ảnh chữ tiếng Việt có dấu, cùng số lượng và cùng font/cỡ chữ với C4, để **chỉ thay đổi một biến là ngôn ngữ**.

**Phạm vi gắn LoRA của C6 và C7:** cả hai chỉ gắn vào các phép chiếu attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) của phần LLM. Vision tower, merger và MLP không gắn ở cả hai nhánh, để C6 và C7 chỉ khác nhau ở chỗ gắn.
- C6: chỉ các layer có head đã chẩn đoán ở C5. Biến thể chặt che gradient ngoài các head đã chọn. Qwen2.5-VL-3B dùng GQA (16 query head, 2 KV head mỗi layer), nên ở `k_proj`/`v_proj` một KV head dùng chung cho 8 query head; biến thể chặt chỉ che được hoàn toàn ở `q_proj` và `o_proj`, phải ghi rõ trong báo cáo.
- C7: cả 36 layer của LLM. Chữ "toàn bộ" nghĩa là toàn bộ layer attention, không phải toàn bộ model.

### 6.1. Trần trên và quy tắc mở rộng (khai báo trước, 17/11/2026)

Mục này được ghi trước khi có kết quả dev của bất kỳ nhánh crop nào trên bộ nhãn mới, để nhánh thêm (nếu có) không bị coi là chọn sau khi thấy số liệu.

**Nhánh O, crop theo bbox chuẩn (chỉ để phân tích, không phải phương pháp):**
- Vùng crop là hợp các khung `evidence_bbox` (với `bbox_mode = "any"` lấy khung đầu tiên), nới ra tâm giữ nguyên cho tới ít nhất kích thước crop đã chọn cho C5, cắt theo mép trang.
- Cùng cách đưa ảnh vào model như C5 (ảnh nguyên + vùng crop), cùng prompt, cùng `max_pixels`. O chỉ khác C5 ở vị trí vùng crop.
- Chỉ chạy trên VisR-Bench tiếng Việt (bộ duy nhất có bbox đáp án). Chạy trên dev để ra quyết định; chạy trên test như một dòng tham chiếu, không đưa vào kiểm định giả thuyết.
- O dùng nhãn nên là trần trên của mọi cách crop cùng kích thước, không phải kết quả của ĐA2.

**Đại lượng dùng để quyết định** (VisR-Bench tiếng Việt, dev, nhóm `short`, tỉ lệ đúng theo LLM-judge, ký hiệu J):
- Δ_O = J(O) − J(B1a), với B1a là baseline tăng `max_pixels` cân số token ảnh;
- Δ_best = max(J(C3), J(C4), J(C5)) − J(B1a).

*Sửa đổi 18/11/2026 (EM → LLM-judge).* Bản ghi ngày 17/11 dùng EM. Kết quả B0 trên dev cho thấy EM không đo được việc đọc đúng trên bộ này: với cùng prompt p2, model trả lời dài hơn đáp án chuẩn ở tiếng Việt (VisR `short`: trung vị 15 từ so với 6 từ của đáp án; 35% câu dài hơn 3 lần), nên EM chỉ 13,7% trong khi 50,6% câu trả lời chứa đáp án (DocVQA: EM 81, chứa 83). Khi đó Δ theo EM chủ yếu đo việc crop có làm model trả lời ngắn lại hay không. Tỉ lệ chứa đáp án cũng không dùng được vì tính đúng cả câu trả lời sai có lẫn chuỗi đáp án. Sửa đổi được ghi khi mới có kết quả B0 trên dev, **trước khi có kết quả dev của B1, O hay bất kỳ nhánh crop nào**; ngưỡng X, Y và bảng quy tắc giữ nguyên. LLM-judge chỉ được dùng khi Cohen's κ với người chấm trên 50 câu `short` của dev (seed cố định) đạt ≥ 0,6; nhãn người được duyệt từng câu từ đề xuất của Codex (chấm chỉ từ chữ, không thấy judge), báo cáo kèm số câu người sửa đề xuất và κ judge – Codex; không đạt thì quay về EM và ghi rõ trong báo cáo. Prompt sinh đáp án p2 giữ nguyên.

*Judge cụ thể (chốt 18/11/2026, trước khi chạy judge trên bất kỳ nhánh nào).* Gemini API gói miễn phí chỉ cho 20 lượt/ngày, không đủ cho khoảng 20–30 nghìn lượt chấm, nên judge chạy cục bộ trên Kaggle bằng `llama.cpp` (`notebooks/judge_local.ipynb`, `src/eval/judge.py`; prompt `j1`, temperature 0, đầu ra JSON giới hạn bằng schema, tắt chế độ suy nghĩ; judge không xem ảnh). Trọng số cố định nên tái lập được; phiên bản llama.cpp (commit) được ghi cùng kết quả.
- Judge chính: Qwen3.8-27B, GGUF `UD-Q4_K_M` (`unsloth/Qwen3.8-27B-GGUF`), chia layer qua 2 T4. Cách xa Qwen2.5 của model đang được chấm nhiều thế hệ.
- Judge đối chứng: Gemma 4 12B-it, GGUF QAT `q4_0` của Google (`google/gemma-4-12B-it-qat-q4_0-gguf`), khác họ; chấm 50 câu mẫu để báo cáo κ giữa hai judge và kiểm thiên vị của judge Qwen.
- Quy tắc chọn: Qwen3.8-27B là judge chính nếu chấm ≤ 2 giây/câu trên 50 câu mẫu; chậm hơn thì đổi vai cho Gemma 4 12B. Judge được chọn vẫn phải đạt κ ≥ 0,6. Không chọn judge theo κ.
- Mốc so sánh báo cáo kèm: luật "câu trả lời chứa đáp án chuẩn" (κ = 0,72 với nhãn người trên 50 câu mẫu). Judge LLM không vượt mốc này thì ghi rõ trong báo cáo.
- Đã loại: Granite 4.2 8B (tiếng Việt không có trong danh sách ngôn ngữ đã kiểm thử), UniRRM-14B (reward model tự sinh tiêu chí và chấm thang 1–5, sinh hàng nghìn token mỗi câu, không hợp chấm đúng/sai theo đáp án).

**Ngưỡng: Y = 8 điểm %, X = 4 điểm %.**
- Dev có 168 câu `short` (bộ nhãn cuối, 18/11/2026; bản ghi 17/11 dự kiến 140–150). Với 15–20% câu đổi kết quả giữa hai cấu hình, sai số chuẩn của hiệu ghép cặp khoảng 3–3,5 điểm %. Y = 8 điểm là khoảng 2,3–2,7 lần sai số chuẩn, và lớn hơn mức chênh test phát hiện được (khoảng 4,5–6,4 điểm với 387 câu `short` ở test, mục 7.1). Dưới mức này, khoảng trống của cả họ phương pháp crop không đủ lớn để một nhánh mới có cơ hội cho kết quả kiểm định được trên test.
- X = 4 điểm = Y/2: các nhánh crop theo attention lấy được dưới một nửa mức trần.

**Quy tắc:**

| Tình huống trên dev | Quyết định |
|---|---|
| Δ_O ≥ 8 và Δ_best < 4 | Khâu định vị là nút thắt. Thêm **C8** (dưới đây) |
| Δ_O ≥ 8 và Δ_best ≥ 4 | Crop theo attention đã lấy được phần đáng kể của trần. Không thêm nhánh; phân tích phần còn hụt bằng tỷ lệ crop chứa bbox |
| Δ_O < 8 | Phóng to không phải nút thắt chính trên tài liệu tiếng Việt. Không thêm nhánh crop; thời gian dành cho C6/C7 và phân tích lỗi (lỗi do dấu, theo `question_type`) |

**C8, Qwen2.5-VL tự khoanh vùng (chỉ chạy khi tình huống 1 xảy ra):**
- Bước 1: hỏi Qwen2.5-VL-3B tọa độ vùng chứa đáp án (đầu ra bbox dạng JSON mà model đã được huấn luyện). Prompt khoanh vùng chốt một lần trước khi chạy dev, không dò prompt.
- Bước 2: crop và trả lời giống hệt C5 (cùng kích thước crop tối thiểu, cùng cách đưa ảnh, cùng prompt p2). Không đọc được bbox thì dùng ảnh nguyên và ghi tỷ lệ lỗi đọc bbox.
- Không cần attention hay backward. Siêu tham số duy nhất được dò trên dev là kích thước crop, cùng lưới với C5.

**Ràng buộc:**
- Quyết định được ghi vào `results/extension_decision.json` (các giá trị Δ_O, Δ_best, số câu, ngày) ngay khi có kết quả dev của C3–C5 và O, dự kiến cuối tuần 7.
- C8 phải chạy được trên dev trước khi chốt cấu hình ở tuần 10; không kịp thì không chạy trên test và ghi rõ trong báo cáo.
- Nhiều nhất một nhánh thêm. So sánh chính vẫn là C5 vs C4. C8 vs C5 và C8 vs B1 thêm vào nhóm so sánh phụ, hiệu chỉnh Holm.
- Ngưỡng X, Y và quy tắc trên không được đổi sau khi đã thấy kết quả dev.

## 7. Dataset

### 7.1. Đánh giá: thiết kế 2 × 2

| | Ảnh cảnh, ít chữ | Tài liệu, dày chữ |
|---|---|---|
| **Tiếng Anh** | TextVQA (tập validation) | DocVQA (tập validation) |
| **Tiếng Việt** | ViTextVQA (tập test) | **VisR-Bench tiếng Việt** (tự xây, có bbox) |

- **Ba bộ công khai:** lấy ngẫu nhiên với seed cố định, gồm 300 câu dev và 500 câu test mỗi bộ, không trùng ảnh/tài liệu giữa dev và test. Ban đầu dev có 100 câu; ngày 18/11/2026 (mới chạy B0, chưa chạy nhánh nào khác) thêm 200 câu dev mỗi bộ, lấy từ các nhóm ngay sau 600 nhóm đầu của cùng thứ tự xáo, nên 500 câu test và 100 câu dev cũ giữ nguyên từng câu (`dev_batch` = `orig` / `ext1`). Lý do: với 100 câu, chênh lệch nhỏ nhất giữa hai nhánh phát hiện được là khoảng 9–12 điểm; 300 câu còn khoảng 5–7 điểm.
- **VisR-Bench tiếng Việt:** 831 câu trên 19 tài liệu, 320 trang (bộ nhãn cuối sau hai vòng duyệt, 18/11/2026). Chia dev/test theo `document_id`, khoảng 30/70, chốt và ghi lại trước khi chạy nhánh nào. Quy tắc (script `src/data_prep/split_visr.py`):
  - duyệt hết 2^19 cách gán tài liệu vào dev, nên kết quả tất định, không cần seed;
  - ràng buộc: phần dev 27–33% số câu (chia theo tài liệu nên không đạt đúng 30%); dev ít nhất 5 tài liệu; không tài liệu nào chiếm quá nửa test;
  - trong các cách thỏa ràng buộc, chọn cách có tổng khoảng cách biến phân toàn phần (TVD) giữa dev và test nhỏ nhất trên ba phân bố: `question_type`, dạng đáp án, cờ dấu; hòa thì lấy cách có phần dev gần 30% hơn;
  - chỉ dùng nhãn, không dùng kết quả model. Chạy một lần sau khi loại bằng chứng đã chốt; cấu hình, phân bố hai phần và 10 cách gán tốt nhất lưu ở `data/qa/visr_bench_vi_v2_split.json`.
  - Lý do chọn 30/70: dev cần đủ câu `short` cho quy tắc mở rộng (mục 6.1; với khoảng 166 câu, sai số chuẩn của Δ_O khoảng 4 điểm, nên Y = 8 xấp xỉ 2 lần sai số chuẩn) và để chọn siêu tham số LoRA. Hạ dev xuống 20% chỉ làm test nhạy hơn khoảng 0,3–0,5 điểm, còn nâng lên 50% làm test kém nhạy hơn khoảng 1 điểm (McNemar, α = 0,05, power 80%). Với khoảng 390 câu `short` ở test, độ lệch nhỏ nhất C5 – C4 phát hiện được là khoảng 4,5–6,4 điểm (tùy tỉ lệ câu bất đồng giữa hai nhánh, 10–20%); đây là giới hạn của bộ dữ liệu, ghi vào báo cáo.
- So sánh giữa các ô dựa trên **mức chênh trước và sau can thiệp** trong cùng một bộ, không so điểm tuyệt đối giữa các bộ. Lý do: Qwen2.5-VL có thể đã học tập train của TextVQA/DocVQA.

### 7.2. Train cho C6, C7

| Bộ | Phần dùng | Nhãn |
|---|---|---|
| ViInfographicVQA | single-image train, lọc câu trích từ ảnh (`image-span`, `multi-span`), ưu tiên loại Text/Table | Gemini sinh, người duyệt |
| ViTextVQA | tập train | Người gán *[cần kiểm tra quy trình]* |
| MTVQA tiếng Việt | tập train (4.011 câu) | Người bản ngữ gán |

- Ô ảnh cảnh tiếng Việt (ViTextVQA) gần miền train của C6/C7; ô tài liệu tiếng Việt là chuyển miền. Phân tích RQ4 dựa trên sự khác biệt này.
- Đang xin thêm dữ liệu cùng miền: phần tiếng Việt của SEA-Vision, Viet-Doc-VQA-II (kiểm 100 mẫu trước khi dùng). Chi tiết ở [Nguon_du_lieu_tieng_Viet.md](Nguon_du_lieu_tieng_Viet.md).

### 7.3. Ảnh chẩn đoán head

Khoảng 1.000 ảnh tiếng Anh và 1.000 ảnh tiếng Việt, tự sinh, dùng cho C4, C5 (đã làm).

### 7.4. Quy định chung

- **Chia theo tài liệu:** không tài liệu nào xuất hiện ở cả dev và test. Tài liệu dev/test không bao giờ dùng để train.
- **Trường dữ liệu:** `document_id`, `page_id`, `image`, `question`, `answer`, `evidence_bbox`, `question_type` (small_detail / table / text, gán bằng luật ở mục 9), `has_diacritic`, `split`.
- **Quy trình tạo (VisR-Bench tiếng Việt):** câu hỏi gốc của VisR-Bench → VLM mạnh sinh nháp đáp án + bbox → người duyệt 100% (vòng 1: 109 câu; vòng 2: 936 câu).
- **Kiểm tra chất lượng:** 2 người gán độc lập trên ≥20% test, báo cáo Cohen's κ (đáp án) và IoU (bbox).

## 8. Mô hình và công nghệ

| Thành phần | Lựa chọn |
|---|---|
| VLM | Qwen2.5-VL-3B (chính); Vintern-1B (baseline tham khảo) |
| Can thiệp crop | Code ViCrop ([mllms_know](https://github.com/saccharomycetes/mllms_know), MIT, đã hỗ trợ Qwen2.5-VL); HAVC **tự cài đặt** trên nền code ViCrop (paper không công bố code) |
| LoRA | PEFT; QLoRA 4-bit nếu thiếu VRAM; chỉ cập nhật adapter |
| Đánh giá | EM, F1, LLM-judge cục bộ (`llama.cpp` trên Kaggle) |

Lưu ý kỹ thuật:
- Lấy attention cần chế độ eager attention; T4 không hỗ trợ bf16 gốc và FlashAttention-2, nên dùng fp16 và giới hạn `max_pixels`.
- Ngoại lệ (từ 17/11/2026): **vision tower chạy fp32**, phần LLM giữ fp16. Với vision tower fp16, 6/409 câu dev bị tràn số ở `visual.blocks.31.mlp.down_proj`; đổi sang fp32 sửa cả 6 câu, tốn thêm khoảng 1,4 GB VRAM. Áp dụng cho mọi nhánh và cho chẩn đoán head (`src/vlm/qwen.py`, `notebooks/fp16_overflow_diag.ipynb`).
- HAVC và `grad_att` cần backward lúc inference, nên tốn VRAM; đo trước trên dev.
- Lúc train LoRA không cần lấy attention weights, nên không bắt buộc eager. Dùng gradient checkpointing và giới hạn `max_pixels`.
- Hạn mức Kaggle khoảng 30 giờ GPU/tuần; đo thời gian một lượt train trước khi lên kế hoạch số cấu hình.

## 9. Đánh giá

| Nhóm | Chỉ số |
|---|---|
| Câu trả lời | EM, token-level F1 (chuẩn hóa Unicode NFC, hoa/thường), LLM-judge có đáp án chuẩn (đo Cohen's κ với người trên ~50 câu) |
| Lỗi do dấu | EM/F1 khi bỏ dấu cả đáp án lẫn dự đoán. Chênh lệch giữ dấu – bỏ dấu là phần lỗi do dấu (chỉ để phân tích, không thay chỉ số chính) |
| Định vị của can thiệp | Tỷ lệ vùng crop chứa bbox đáp án; IoU giữa vùng crop và bbox |
| Cơ chế HAVC | Entropy của từng head, số head qua lọc, diện tích vùng crop; so giữa ảnh cảnh và tài liệu |
| Chẩn đoán head | Độ trùng attention với vùng chữ (head chọn bằng tiếng Anh vs tiếng Việt), riêng cho vùng dấu thanh |
| Chi phí | Thời gian/câu, VRAM đỉnh, số forward/backward pass; với C6/C7 thêm số tham số train, thời gian train, VRAM đỉnh lúc train |

**Prompt sinh đáp án (chốt 11/2026, trước khi chạy các nhánh trên bộ nhãn mới; code: `src/vlm/prompts.py`):**
- Một prompt cho mọi câu và mọi nhánh, nối sau câu hỏi: *"Trả lời câu hỏi dựa vào tài liệu. Chép nguyên văn từ tài liệu, ngắn nhất mà vẫn đủ ý. Nếu câu hỏi yêu cầu nhiều ý (liệt kê, các bước, các điều kiện, các thành phần), nêu đủ mọi ý, mỗi ý ngắn gọn, ngăn cách bằng dấu chấm phẩy. Chỉ ghi đáp án, không giải thích."*
- `max_new_tokens = 384`, giải mã tham lam (greedy). Đáp án nhiều ý có trung vị 41 từ, 10% dài hơn 139 từ.
- Không chọn prompt theo `question_type` hay dạng đáp án, vì đó là thông tin nhãn.
- Bản trước (p1: *"Trả lời ngắn gọn bằng một cụm từ hoặc một câu."*, 64 token) chỉ dùng cho C3 trên dev cũ; mọi nhánh chạy lại với bản p2.
- Bộ tiếng Anh (TextVQA, DocVQA) dùng bản dịch tiếng Anh của cùng câu chỉ dẫn; cố định trước khi chạy.
- C6/C7 train với cùng câu chỉ dẫn nối sau câu hỏi.

**Dạng đáp án (tự gán từ đáp án chuẩn, `src/eval/score.py`):**
- `list`: từ 2 ý trở lên, ngăn bằng "; ";
- `long`: một ý dài hơn 25 từ (hoặc người duyệt đánh dấu);
- `short`: còn lại.

Báo cáo kết quả tách theo dạng đáp án; câu `list` chấm thêm độ phủ theo ý (`item_cover`). Phân tích chính của RQ2 (crop) dựa trên nhóm `short`, vì đáp án dài trải trên vùng lớn mà crop không nhắm tới.

**Trục độ khó và `question_type` (VisR-Bench tiếng Việt; chốt 18/11/2026, trước khi chạy bất kỳ nhánh nào trên dev):**

Câu hỏi lấy từ VisR-Bench, không được soạn theo mức độ khó, nên độ khó được tính sau từ nhãn đã duyệt (bbox, đáp án, kích thước ảnh), không gán tay. Các trục chỉ dùng để chia nhóm khi báo cáo và để cân dev/test; không dùng để chọn prompt, cấu hình hay nhánh.

| Trục | Cách tính | Nhóm | Áp dụng |
|---|---|---|---|
| Kích thước đích | Diện tích vùng bằng chứng / diện tích trang. `bbox_mode = "all"`: hộp bao các khung; `"any"`: khung nhỏ nhất | nhỏ < 0,5% · vừa 0,5–5% · lớn > 5% | Mọi câu; phân tích chính trên `short` |
| Độ phân giải hiệu dụng | Chiều cao khung thấp nhất × s, với s = min(1, √(`max_pixels` / (W·H))), `max_pixels` = 1024·28·28. Xấp xỉ chiều cao một dòng chữ trong ảnh model nhận (có lẫn phần đệm của khung) | thấp < 15 px · vừa 15–25 px · cao > 25 px | Chỉ `short` |
| Dấu | Đáp án có ít nhất một ký tự mang dấu tiếng Việt (dấu thanh, dấu mũ/móc, đ) | có / không | Mọi câu |
| Loại bằng chứng | `evidence_type` của nháp Gemini; các câu `table` và các câu Gemini bỏ trống được Codex kiểm trên ảnh, người duyệt các câu Codex không đồng ý với Gemini hoặc không chắc | text · table · figure | Mọi câu |
| Dạng đáp án | `answer_form` (mục trên) | short · list · long | Mọi câu |

Lý do chọn trục: kích thước đích là trục ViCrop dùng để chỉ ra MLLM kém với chi tiết nhỏ (*ngưỡng cụ thể của ViCrop chưa đối chiếu*), nên kết quả tiếng Việt so được với phát hiện gốc. Độ phân giải hiệu dụng đo trực tiếp hệ quả của việc thu ảnh: trang trung vị 3,9 MP bị thu về khoảng 0,8 MP, một token ảnh của Qwen2.5-VL phủ 28 px, trong khi dấu thanh chỉ cao vài pixel.

`question_type` gán bằng luật, theo thứ tự ưu tiên, mỗi câu một nhãn:
1. `table`: loại bằng chứng là bảng (đã kiểm như trên);
2. `small_detail`: còn lại, dạng `short`, kích thước đích nhỏ (< 0,5%);
3. `text`: còn lại.

`diacritic` không còn là một giá trị của `question_type` mà thành cờ `has_diacritic` (trục "Dấu"). Lý do: 83% đáp án `short` hiện có mang dấu (183/221), nên làm một loại riêng không tách được nhóm nào. Lỗi do dấu được đo trên mọi câu bằng chênh lệch điểm giữ dấu – bỏ dấu (bảng chỉ số ở trên).

Phân bố trên 418 câu đã duyệt (17/11/2026, để tham khảo; tính lại khi có bộ nhãn cuối): `short` 221 câu, trong đó 82 câu kích thước đích nhỏ; độ phân giải hiệu dụng thấp / vừa / cao = 37 / 88 / 96 câu; loại bằng chứng text 335, table 64, figure 10.

Giới hạn cần nêu trong báo cáo:
- Ít câu hỏi về hình (39/831 câu figure, 5 câu ở dev): dataset mô tả là "trang tài liệu dày chữ và bảng", không phải tài liệu chứa hình. Nhóm figure chỉ báo cáo mô tả, không kiểm định.
- Ít tài liệu và lệch: 19 tài liệu, 5 tài liệu lớn nhất chiếm 64% số câu. Khoảng tin cậy báo cáo thêm bằng bootstrap theo tài liệu (`src/eval/stats.py`); báo cáo cả kết quả khi bỏ từng tài liệu lớn (leave-one-document-out) cho so sánh chính.
- Nhóm có dưới 30 câu ở test chỉ báo cáo mô tả, không kiểm định riêng.

**Kiểm định thống kê:**
- So sánh ghép cặp giữa các nhánh trên cùng câu hỏi bằng **McNemar** (đúng/sai) và **paired bootstrap** (F1, LLM-judge), báo cáo khoảng tin cậy 95%.
- So sánh chính (C5 vs C4) khai báo trước; các so sánh phụ hiệu chỉnh **Holm**.
- Ước lượng (power 80%, α = 0,05): với 10–20% câu đổi kết quả giữa hai cấu hình, 387 câu `short` ở test VisR phát hiện được chênh lệch khoảng 4,5–6,4 điểm %; 500 câu test mỗi bộ công khai khoảng 4,0–5,6 điểm %.

## 10. Đóng góp dự kiến

1. **Phương pháp:** biến thể HAVC với tác vụ chẩn đoán head theo ngôn ngữ đích (tiếng Việt có dấu), kèm so sánh có kiểm soát với ViCrop và HAVC gốc (RQ2, RQ3).
2. **Phân tích:** crop theo attention còn tác dụng tới đâu khi đổi loại ảnh và ngôn ngữ (RQ1); head/layer nào của VLM định vị chữ tiếng Việt, khác gì so với chữ tiếng Anh.
3. **So sánh training-free với fine-tune nhắm head** khi không có dữ liệu train cùng miền (RQ4).
4. **Tài nguyên:** dataset dev/test tài liệu tiếng Việt có bbox và nhãn loại câu hỏi; bộ ảnh chẩn đoán tiếng Việt; code tái hiện HAVC; code LoRA nhắm head.

## 11. Kế hoạch thực hiện (14 tuần)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 1–4 (đã xong) | Đọc tài liệu; pilot ViCrop; xây quy trình gán nhãn, duyệt vòng 1; quét C3 trên dev cũ; chẩn đoán head EN/VI | Báo cáo tuần 1–4 |
| 5–6 (đang làm) | Duyệt xong vòng 2, chia dev/test, gán `question_type`; script kiểm định thống kê; tải và lấy mẫu TextVQA, DocVQA, ViTextVQA; chạy B0 trên 4 bộ (dev) | Dataset v2; bảng B0 |
| 6–7 | B1; C3, C4, C5, R trên dev 4 bộ; O (crop theo bbox chuẩn) trên dev VisR-Bench; đo VRAM/thời gian của HAVC; áp quy tắc mở rộng mục 6.1 | Kết quả RQ1–RQ3 trên dev; `extension_decision.json` |
| 7–9 | Chuẩn bị dữ liệu train; chạy thử QLoRA (VRAM, thời gian); train C6, C7; chọn siêu tham số trên dev; C8 nếu quy tắc 6.1 kích hoạt | C6, C7 (và C8 nếu có) chạy được trên dev |
| 10 | Chốt toàn bộ cấu hình và ghi lại (khai báo trước khi chạy test) | Cấu hình cuối |
| 11 | Chạy **một lần** tất cả nhánh trên test 4 bộ | Bảng kết quả |
| 12 | Kiểm định thống kê, phân tích lỗi theo loại câu hỏi, phân tích head | Phân tích |
| 13 | Viết báo cáo | Báo cáo |
| 14 | Chuẩn bị bảo vệ | Đồ án hoàn chỉnh |

**Tiến độ đến 18/11/2026 (tuần 5–6):**

| Việc | Trạng thái |
|---|---|
| Lấy mẫu 3 bộ công khai (`src/data_prep/sample_public.py`) | Xong: TextVQA, DocVQA, ViTextVQA, mỗi bộ 300 dev (100 ban đầu + 200 thêm ngày 18/11) + 500 test, một câu mỗi ảnh/tài liệu, dev và test không chung ảnh/tài liệu |
| Bộ đọc chung 4 bộ đánh giá (`src/data_prep/eval_sets.py`) | Xong; VisR-Bench đọc từ `visr_bench_vi_v2.jsonl`, kèm các trục độ khó |
| Script kiểm định (`src/eval/stats.py`) | Xong: McNemar chính xác, paired bootstrap (theo câu và theo tài liệu), Holm cho so sánh phụ |
| Prompt sinh đáp án p2, dạng đáp án, `item_cover` | Xong (mục 9) |
| Duyệt vòng 2 VisR-Bench tiếng Việt | Xong (18/11/2026): duyệt tay đủ 936 câu (Codex kiểm và đề xuất, người quyết định từng câu); giữ 831 câu (769 đúng, 62 sửa), bỏ 105 câu (46 cần nhiều trang, 29 trang không có đáp án, 28 câu hỏi mơ hồ, 1 đáp án ở trang khác, 1 lý do khác) |
| Chia dev/test VisR-Bench, gán `question_type` | Xong (18/11/2026), `data/qa/visr_bench_vi_v2.jsonl`: dev 251 câu (30,2%, 8 tài liệu), test 580 câu (11 tài liệu); `question_type` small_detail / table / text = 29 / 46 / 176 (dev) và 64 / 106 / 410 (test); tổng TVD 0,010. Loại bằng chứng: Codex kiểm 187 câu, người duyệt 55 câu |
| B0 trên dev 4 bộ | Chạy lại với vision tower fp32 (18/11/2026): hết 6 câu suy biến, điểm không giảm (`results/B0_base/README.md`). Xong cả 4 bộ dev (19/11/2026): VisR 251 câu; ba bộ công khai 300 câu mỗi bộ, 100 câu cũ chạy lại cho câu trả lời giống hệt từng ký tự (tái lập 300/300). Lượt fp16 cũ giữ ở `results/B0_base/superseded_fp16_vision/` |
| Chẩn đoán head EN/VI (C4, C5) | Xong, chạy lại với vision tower fp32 (17/11/2026). Danh sách expert head **trùng hoàn toàn** với lượt fp16 (EN 35/35, VI 44/44, cùng thứ tự top-8); tỉ lệ đọc đúng token VI tăng nhẹ 99,17% → 99,46%, EN giữ 99,68%. Chi phí: 1,24–1,26 s/ảnh (gấp khoảng 2 lần fp16), VRAM đỉnh 6,66 GB. Code inference HAVC đã có |
| Dữ liệu train C6/C7 | Đã tải ViInfographicVQA; đã soạn thư xin dữ liệu ([Thu_xin_du_lieu.md](Thu_xin_du_lieu.md)) |

Việc tiếp theo: chạy judge cục bộ (`notebooks/judge_local.ipynb`): chấm 50 câu mẫu, đo κ, chấm B0 dev → B1, C3–C5, R, O trên dev.

## 12. Rủi ro và phương án dự phòng

| Rủi ro | Ảnh hưởng | Phương án |
|---|---|---|
| Lấy attention / backward vượt VRAM T4 | Không chạy được C3–C5 | Dùng ViCrop `rel_att` (chỉ forward); giảm `max_pixels`; dùng Kaggle 2×T4; bỏ thành phần gradient của HAVC (ghi rõ sai khác); chuyển sang FOCUS (không cần gradient) |
| Tự cài HAVC sai khác paper | C4 không tái hiện được | Kiểm tra trên TextVQA một tập nhỏ, so với số trong paper; báo cáo rõ sai khác |
| Can thiệp không cải thiện trên tiếng Việt | RQ2/RQ3 cho kết quả âm | Vẫn báo cáo được; nhánh O (crop theo bbox chuẩn) tách lỗi định vị khỏi lỗi đọc; thiết kế 2 × 2 và phân tích cơ chế (entropy, số head qua lọc) giải thích nguyên nhân; nhánh thêm C8 chỉ theo quy tắc khai báo trước ở mục 6.1 |
| QLoRA vượt VRAM hoặc quá chậm trên T4 | Không chạy được C6/C7 | Giảm `max_pixels`, rank, độ dài chuỗi; gradient checkpointing; giảm số mẫu train; dùng Vintern-1B nếu Qwen 3B không vừa (ghi rõ) |
| Dữ liệu train lệch miền | LoRA không chuyển được sang tài liệu | Là một kết quả của RQ4, vẫn báo cáo; xin thêm dữ liệu cùng miền (SEA-Vision, Viet-Doc-VQA-II) |
| Hết hạn mức Kaggle | Chậm tiến độ | Đo trước thời gian mỗi lượt; ưu tiên C5, C4, C6, C7; giảm số cấu hình dò |
| Annotation chậm | Không đủ câu test | Ưu tiên test hơn dev; giữ đủ nhóm chi tiết nhỏ; nếu dưới 300 câu thì báo cáo là kết quả khảo sát, không kết luận mạnh |
| LLM-judge thiếu ổn định | Đánh giá kém tin cậy | Đáp án chuẩn do người duyệt; đo κ với người; báo cáo kèm EM/F1 |

## 13. Kết quả đầu ra

- Dataset dev/test tài liệu tiếng Việt có bbox và nhãn loại câu hỏi; bộ ảnh chẩn đoán head tiếng Việt.
- Code các nhánh B0, B1, C3–C7 và R; code tái hiện HAVC; code LoRA nhắm head; script đánh giá và kiểm định thống kê.
- Bảng kết quả 2 × 2, phân tích theo loại câu hỏi, phân tích head/layer, case study crop đúng/sai.
- Báo cáo ĐA2.

## 14. Liên kết với khóa luận

Khóa luận mở rộng trên kết quả của ĐA2:

- **Truy xuất trên tài liệu nhiều trang:** OCR + LLM (A), RAG văn bản + VLM (B), Multimodal RAG có sẵn (C1: ColVintern-1B + Vintern-1B; C2: ColQwen + Qwen2.5-VL-3B), rồi đặt C5/C6 lên trên nhánh truy xuất tốt nhất. Đo Recall@K, MRR cấp trang.
- **LoRA với dữ liệu cùng miền**, nếu xin được dữ liệu trang tài liệu tiếng Việt có nhãn người (SEA-Vision, ...).
- **Agentic RAG (tùy chọn):** đưa crop theo attention thành một tool mà agent tự quyết định gọi, so với crop cố định ở C5.
