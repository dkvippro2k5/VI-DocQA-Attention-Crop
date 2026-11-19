# BÁO CÁO TIẾN ĐỘ ĐỒ ÁN 2 (SAU TUẦN 4)

**Đề tài:** Vision-Language Model trong hỏi đáp tài liệu tiếng Việt
**Sinh viên:** Dương Văn Kiên
**MSSV:** 20233857
**GVHD:** Dr. Đỗ Thị Ngọc Diệp
**Thời gian:** 13/11/2026 – 19/11/2026 (nối tiếp báo cáo tuần 4; chưa phải báo cáo tuần 5)
**Nội dung:** Điều chỉnh scope theo góp ý, sửa lỗi tràn số, chốt bộ dữ liệu và cách chia dev/test, chạy baseline B0 trên 4 bộ, đổi chỉ số chính sang LLM-judge

---

## 1. Tóm tắt

- **Scope đã điều chỉnh** theo góp ý sau tuần 4 (GVHD đồng ý ngày 15/11): ĐA2 chỉ còn bài toán đọc **một trang**; phần truy xuất nhiều trang chuyển sang khóa luận. Đánh giá theo thiết kế 2 × 2 (Anh/Việt × ảnh cảnh/tài liệu); thêm hai nhánh LoRA C6, C7. So sánh chính khai báo trước vẫn là C5 vs C4.
- **Sửa một lỗi kỹ thuật:** vision tower của Qwen2.5-VL chạy fp16 bị tràn số, làm 6/409 câu dev sinh chuỗi "!!!!…". Chuyển vision tower sang fp32 thì hết lỗi. Chạy lại chẩn đoán head với fp32: danh sách head **không đổi**.
- **Bộ dữ liệu VisR-Bench tiếng Việt đã chốt:** duyệt tay đủ 936 câu vòng 2; giữ **831 câu** trên 19 tài liệu, 320 trang. Chia dev/test theo tài liệu: dev 251 câu, test 580 câu.
- **Ba bộ công khai** (TextVQA, DocVQA, ViTextVQA): mỗi bộ 300 câu dev và 500 câu test.
- **Baseline B0 đã chạy xong trên cả 4 bộ dev**, tái lập được từng ký tự. Phát hiện chính: ở tiếng Việt, model trả lời dài hơn đáp án chuẩn nhiều, nên EM không đo đúng khả năng đọc. Trên VisR-Bench, EM chỉ 13,7% trong khi 50,6% câu trả lời có chứa đáp án.
- **Chỉ số chính đổi từ EM sang LLM-judge,** ghi là sửa đổi trước khi có kết quả của bất kỳ nhánh can thiệp nào. Judge chạy cục bộ trên Kaggle; đang chạy, chưa có kết quả.

## 2. Điều chỉnh scope

Theo ba góp ý sau tuần 4 (tách hai loại dữ liệu; chạy model gốc trước khi cải tiến; tự cài HAVC thì có thể fine-tune), scope được sửa như sau. Chi tiết ở `docs/Giai_dap_gop_y_GVHD_tuan_4.md` và `docs/De_xuat_scope_Do_an_2.md`.

*Bảng 1. Các nhánh thí nghiệm sau điều chỉnh*

| Nhánh | Nội dung | Huấn luyện |
|---|---|---|
| B0 | Model gốc, ảnh nguyên | Không |
| B1 | Tiền xử lý: tăng `max_pixels` cân số token ảnh với nhánh crop (B1a), xử lý ảnh (B1b) | Không |
| C3 | ViCrop (`rel_att`) | Không |
| C4 / C5 | HAVC, head chọn bằng ảnh chữ tiếng Anh / tiếng Việt có dấu | Không |
| R | Crop cùng kích thước tại vị trí ngẫu nhiên | Không |
| O | Crop theo khung bằng chứng chuẩn (trần trên, chỉ để phân tích) | Không |
| C6 | LoRA chỉ ở các layer có head đã chẩn đoán | LoRA, phần attention của LLM |
| C7 | LoRA ở cả 36 layer attention | LoRA, phần attention của LLM |
| C8 | Qwen tự khoanh vùng rồi crop (chỉ chạy nếu quy tắc mở rộng kích hoạt) | Không |

*Bảng 2. Bốn bộ đánh giá của thiết kế 2 × 2*

| | Ảnh cảnh, ít chữ | Tài liệu, dày chữ |
|---|---|---|
| **Tiếng Anh** | TextVQA (validation) | DocVQA (validation) |
| **Tiếng Việt** | ViTextVQA (test) | VisR-Bench tiếng Việt (tự xây, có khung bằng chứng) |

So sánh giữa các ô dựa trên **mức chênh trước và sau can thiệp** trong cùng một bộ, không so điểm tuyệt đối giữa các bộ, vì Qwen2.5-VL có thể đã học tập train của TextVQA và DocVQA.

Hai điểm được khai báo trước khi có kết quả nhánh can thiệp nào:
- **Quy tắc mở rộng (mục 6.1 của scope):** so trần trên O và nhánh crop tốt nhất với B1a trên dev VisR-Bench, nhóm câu đáp án ngắn. Nếu O hơn B1a ít nhất Y = 8 điểm % mà các nhánh crop theo attention lấy được dưới X = 4 điểm %, thì thêm C8 (khâu định vị là nút thắt). Thêm tối đa một nhánh.
- **Cách chọn cấu hình:** mỗi nhánh chọn **một cấu hình chung** cho cả 4 bộ trên dev gộp (khoảng 1.150 câu), không chọn riêng từng bộ.

## 3. Lỗi tràn số fp16 ở vision tower

Lượt B0 đầu tiên (17/11, 409 câu dev) có 6 câu sinh chuỗi "!!!!…". Chẩn đoán cho thấy giá trị bị tràn fp16 ở `visual.blocks.31.mlp.down_proj`, lớp cuối của vision tower. T4 không hỗ trợ bf16, nên chọn cách chạy **vision tower ở fp32, phần LLM giữ fp16**. Cách này áp dụng cho mọi nhánh và cho chẩn đoán head.

*Bảng 3. B0 trên 100 câu dev mỗi bộ: vision fp16 và fp32*

| Bộ | EM fp16 → fp32 | F1 fp16 → fp32 | Câu suy biến | Câu trả lời giống hệt |
|---|---|---|---|---|
| TextVQA | 63,0 → 65,0 | 73,2 → 76,1 | 2 → 0 | 96/100 |
| DocVQA | 81,0 → 81,0 | 86,9 → 86,9 | 0 → 0 | 100/100 |
| ViTextVQA | 25,0 → 25,0 | 51,5 → 51,6 | 4 → 0 | 91/100 |

Chẩn đoán head chạy lại với vision fp32 cho **cùng danh sách head chuyên gia** (35/35 head tiếng Anh, 44/44 head tiếng Việt, cùng thứ tự 8 head đầu). Tỉ lệ đọc đúng token tiếng Việt tăng nhẹ từ 99,17% lên 99,46%; tiếng Anh giữ 99,68%. Chi phí tăng từ 0,59 lên 1,24 giây/ảnh, VRAM đỉnh từ 4,6 lên 6,66 GB. Như vậy kết quả chẩn đoán head ở báo cáo tuần 4 vẫn dùng được. Kết quả cũ chạy với fp16 được giữ riêng trong `superseded_fp16_vision/`, không dùng để so sánh nhánh.

## 4. Dữ liệu

### 4.1. Duyệt xong vòng 2 VisR-Bench tiếng Việt

Quy trình hai lớp: Codex (một LLM khác) kiểm tra và **đề xuất**, người duyệt **quyết định từng câu**, không dùng nhãn LLM chưa duyệt.
- **Nhãn người đã duyệt trước đó:** Codex chỉ gắn cờ các câu chắc chắn sai. 52 câu được duyệt lại: 31 câu trước đây giữ nay bỏ, 2 câu trước đây bỏ nay giữ lại (có sửa đáp án), 5 câu sửa đáp án, 14 câu giữ nguyên.
- **578 câu chưa duyệt:** Codex đề xuất đáp án và khung bằng chứng, người duyệt xem từng câu trên một trang HTML. Người duyệt đồng ý giữ/bỏ với Codex ở cả 578 câu; giữ nguyên khung Codex ở 373 câu, tự sửa hoặc vẽ lại khung ở 148 câu.

*Bảng 4. Bộ nhãn cuối (936 câu đã duyệt)*

| Hạng mục | Số câu |
|---|---|
| Giữ lại | **831** (769 giữ nguyên đáp án nháp, 62 sửa) |
| Bỏ: cần nhiều trang | 46 |
| Bỏ: trang không có đáp án | 29 |
| Bỏ: câu hỏi mơ hồ hoặc sai | 28 |
| Bỏ: đáp án ở trang khác / lý do khác | 1 / 1 |
| Tài liệu / trang | 19 / 320 |

105/936 câu (11%) không trả lời được từ đúng một trang, trong đó 47 câu cần trang khác hoặc nhiều trang. Các câu này được giữ riêng kèm lý do bỏ, dùng cho phần truy xuất nhiều trang ở khóa luận.

### 4.2. Trục độ khó và loại câu hỏi

Câu hỏi của VisR-Bench không được soạn theo mức độ khó, nên độ khó được **tính từ nhãn** (khung bằng chứng, kích thước ảnh, đáp án), không gán tay. Quy tắc chốt ngày 18/11, trước khi chạy nhánh nào trên dev.

*Bảng 5. Các trục độ khó*

| Trục | Cách tính | Nhóm |
|---|---|---|
| Kích thước đích | Diện tích vùng bằng chứng / diện tích trang | nhỏ < 0,5% · vừa 0,5–5% · lớn > 5% |
| Độ phân giải hiệu dụng (chỉ câu đáp án ngắn) | Chiều cao khung thấp nhất sau khi ảnh bị thu theo `max_pixels` | thấp < 15 px · vừa 15–25 px · cao > 25 px |
| Dấu | Đáp án có ký tự mang dấu tiếng Việt | có / không |
| Loại bằng chứng | Chữ chạy / bảng / hình | text · table · figure |
| Dạng đáp án | Ngắn / nhiều ý / dài | short · list · long |

Loại câu hỏi (`question_type`) gán bằng luật theo thứ tự ưu tiên: `table` (bằng chứng là bảng) → `small_detail` (đáp án ngắn, đích nhỏ) → `text`. Loại "dấu" được tách thành cờ riêng `has_diacritic`, vì 75% đáp án ngắn có dấu, làm thành một loại riêng thì không phân biệt được nhóm nào. Loại bằng chứng lấy từ nháp Gemini; 187 câu (140 câu ghi bảng và 47 câu bỏ trống) được Codex kiểm trên ảnh, người duyệt 55 câu cần xem.

*Bảng 6. Phân bố trên 831 câu*

| Trục | Phân bố |
|---|---|
| Dạng đáp án | short 555 · list 227 · long 49 |
| Loại câu hỏi | text 586 · table 152 · small_detail 93 |
| Loại bằng chứng | text 640 · table 152 · figure 39 |
| Kích thước đích (câu short) | nhỏ 168 · vừa 346 · lớn 41 |
| Độ phân giải hiệu dụng (câu short) | thấp 9 · vừa 129 · cao 417 |
| Có dấu | 682 / 831 |

### 4.3. Chia dev/test

Chia theo tài liệu, khoảng 30/70. Với 19 tài liệu, script duyệt hết 2^19 cách gán, nên kết quả tất định, không cần seed.
- **Ràng buộc:** dev chiếm 27–33% số câu, dev có ít nhất 5 tài liệu, không tài liệu nào chiếm quá nửa test.
- **Tiêu chí:** trong 37.771 cách thỏa ràng buộc, chọn cách mà dev và test lệch nhau ít nhất về loại câu hỏi, dạng đáp án và cờ dấu. Chỉ dùng nhãn, không dùng kết quả model.

*Bảng 7. Kết quả chia*

| | Dev | Test |
|---|---|---|
| Câu / tài liệu | 251 (30,2%) / 8 | 580 / 11 |
| small_detail · table · text | 29 · 46 · 176 | 64 · 106 · 410 |
| short · list · long | 168 · 69 · 14 | 387 · 158 · 35 |
| Có dấu | 82,1% | 82,1% |

**Lý do chọn 30/70:**
- Dev cần đủ câu đáp án ngắn cho quy tắc mở rộng: với 168 câu, sai số chuẩn của chênh lệch ghép cặp khoảng 3–3,5 điểm %, nên Y = 8 vào khoảng 2,5 lần sai số chuẩn.
- Test giữ đủ lớn cho so sánh chính.
- Hạ dev xuống 20% chỉ làm test nhạy hơn khoảng 0,3–0,5 điểm. Nâng dev lên 50% làm test kém nhạy hơn khoảng 1 điểm.

### 4.4. Ba bộ công khai

Lấy mẫu ngẫu nhiên với seed cố định, mỗi ảnh hoặc tài liệu một câu (để các câu gần độc lập), dev và test không chung ảnh/tài liệu.
- Ban đầu mỗi bộ có 100 câu dev và 500 câu test.
- Ngày 18/11 (mới chạy B0) thêm **200 câu dev mỗi bộ**, lấy từ các nhóm ngay sau 600 nhóm đầu của cùng thứ tự xáo. 500 câu test và 100 câu dev cũ giữ nguyên từng câu.
- Lý do: với 100 câu, chênh lệch nhỏ nhất giữa hai nhánh phát hiện được là khoảng 9–12 điểm; 300 câu còn khoảng 5–7 điểm.

## 5. Baseline B0

Qwen2.5-VL-3B-Instruct, LLM fp16, vision fp32, `max_pixels` = 1024·28·28, prompt chung p2, giải mã tham lam, tối đa 384 token. Chạy trên Kaggle T4 x2, ghim phiên bản thư viện (`transformers` 5.20.2, torch 2.11.0).

*Bảng 8. B0 trên dev*

| Bộ | n | EM | Chứa đáp án | F1 | F1 bỏ dấu − F1 | Số từ trả lời / đáp án (trung vị) | Trả lời dài > 3 lần đáp án |
|---|---|---|---|---|---|---|---|
| TextVQA | 300 | 62,7 | 78,3 | 72,8 | +0,3 | 1 / 1 | 8% |
| DocVQA | 300 | 82,0 | 86,0 | 88,3 | +0,0 | 2 / 2 | 3% |
| ViTextVQA | 300 | 25,7 | 50,7 | 51,4 | +3,3 | 4 / 3 | 17% |
| VisR-Bench tiếng Việt (short) | 168 | 13,7 | 50,6 | 44,6 | +0,6 | 15 / 6 | 35% |

*Bảng 9. B0 trên dev VisR-Bench, nhóm đáp án ngắn, theo loại câu hỏi*

| Loại câu hỏi | n | EM | Chứa đáp án | F1 |
|---|---|---|---|---|
| small_detail | 29 | 27,6 | 65,5 | 45,6 |
| table | 40 | 12,5 | 22,5 | 28,3 |
| text | 99 | 10,1 | 57,6 | 51,0 |

Nhận xét (chỉ là quan sát trên dev, chưa qua kiểm định):
- **Tái lập:** 100 câu cũ mỗi bộ chạy lại cho câu trả lời giống hệt từng ký tự (300/300). Không câu nào suy biến.
- **Model trả lời dài khi làm việc bằng tiếng Việt.** Với cùng câu chỉ dẫn, ở tiếng Anh model trả lời gọn (DocVQA: EM 82,0, chứa đáp án 86,0). Ở tiếng Việt model chép cả câu, có khi cả điều khoản, nên EM thấp xa so với tỉ lệ chứa đáp án.
- **Bảng là nhóm khó nhất, và đó là lỗi đọc thật:** đọc nhầm ký tự trong tên hay số (ví dụ "NHAN MY" thành "NAM MY"). Tuy nhiên, 34/40 câu bảng ở dev thuộc cùng một tài liệu, nên chưa tách được ảnh hưởng của "bảng" khỏi ảnh hưởng của tài liệu đó.
- **Chi tiết nhỏ không phải điểm yếu rõ của B0 ở đây:** nhóm small_detail có tỉ lệ chứa đáp án cao nhất (65,5%). Phần lớn câu đáp án ngắn ở độ phân giải hiệu dụng cao.
- **Lỗi do dấu nhỏ:** F1 bỏ dấu chỉ cao hơn F1 giữ dấu 0,6 điểm trên VisR-Bench, và 3,3 điểm trên ViTextVQA. Ở độ phân giải này, model gốc hiếm khi chỉ sai mỗi dấu. Điều này làm giảm kỳ vọng về chênh lệch C5 – C4 (mục 7).

## 6. Đổi chỉ số chính sang LLM-judge

**Lý do:**
- EM đo việc model trả lời có ngắn gọn đúng khuôn hay không, hơn là đo việc model đọc đúng (mục 5). Nếu giữ EM, mức tăng của một nhánh crop có thể chỉ phản ánh việc crop làm model trả lời ngắn lại.
- Tỉ lệ "chứa đáp án" cũng không dùng được, vì nó tính đúng cả câu trả lời sai có lẫn chuỗi đáp án. Ví dụ: đáp án "05 chỉ vàng 9999", model trả lời "Số giải đặc biệt (05 chỉ vàng 9999) đã trúng: 0".

**Sửa đổi:** chỉ số chính cho so sánh chính và cho quy tắc mở rộng là **tỉ lệ đúng theo LLM-judge có đáp án chuẩn**, trên nhóm đáp án ngắn của VisR-Bench. EM, F1 và tỉ lệ chứa đáp án vẫn báo cáo kèm. Sửa đổi được ghi khi mới có kết quả B0, trước khi có kết quả dev của B1, O hay bất kỳ nhánh crop nào; ngưỡng X, Y và prompt sinh đáp án giữ nguyên.

**Judge:**
- Judge nhận câu hỏi, đáp án chuẩn và câu trả lời, trả về đúng/sai kèm một câu lý do; không xem ảnh.
- Tiêu chí cố định:
  - được phép trả lời dài nếu đáp án chuẩn rõ ràng là câu trả lời;
  - sai nếu đưa nhiều ứng viên mâu thuẫn, thiếu ý, hoặc từ chối;
  - **sai dấu tiếng Việt tính là từ khác**, giữ đúng quy tắc không bỏ dấu của chỉ số chính.
- Gemini API gói miễn phí chỉ cho 20 lượt/ngày, không đủ cho khoảng 20–30 nghìn lượt chấm. Vì vậy judge chạy **cục bộ trên Kaggle** bằng `llama.cpp`, với trọng số cố định nên tái lập được.
  - **Judge chính:** Qwen3.8-27B (bản lượng tử 4-bit, chia qua 2 T4). Cách xa Qwen2.5 của model đang được chấm nhiều thế hệ.
  - **Judge đối chứng:** Gemma 4 12B (bản QAT 4-bit của Google), khác họ, dùng để kiểm judge Qwen có thiên vị không.
  - **Quy tắc chọn, ghi trước khi chạy:** Qwen3.8-27B là judge chính nếu chấm ≤ 2 giây/câu, chậm hơn thì Gemma làm judge chính. Không chọn judge theo κ.

**Kiểm độ tin cậy:**
- 50 câu đáp án ngắn của dev, lấy theo seed cố định. Codex đề xuất đúng/sai chỉ từ chữ, không thấy judge; người duyệt xem lại từng câu. Kết quả: người đồng ý cả 50 đề xuất, 25 đúng và 25 sai.
- Judge chỉ được dùng khi Cohen's κ với nhãn người đạt ≥ 0,6; không đạt thì quay về EM.
- Mốc so sánh: luật đơn giản "câu trả lời chứa đáp án" đã đạt κ = 0,72 trên 50 câu này. Judge LLM cần vượt mốc này mới có giá trị.

**Tình trạng:** notebook judge đang chạy trên Kaggle; chưa có κ và chưa có điểm judge của B0.

## 7. Tình trạng các nhánh

*Bảng 10. Tình trạng*

| Nhánh | Tình trạng |
|---|---|
| B0 | Xong trên dev 4 bộ |
| Judge | Đang chạy (chấm 50 câu mẫu, tính κ, chấm B0 dev) |
| B1a, B1b | Chưa chạy |
| C3 | Code có sẵn; kết quả cũ (prompt p1, dev cũ) đã chuyển sang thư mục lưu trữ, sẽ chạy lại với cấu hình mới |
| C4, C5 | Head đã chẩn đoán lại với fp32; code suy luận có sẵn, chưa đo VRAM |
| R, O | Chưa chạy |
| C6, C7 | Đang chuẩn bị dữ liệu train |
| C8 | Chỉ chạy nếu quy tắc mở rộng kích hoạt |

## 8. Hạn chế và rủi ro

- **So sánh chính có thể cho kết quả không có ý nghĩa thống kê.** Với 387 câu đáp án ngắn ở test, độ lệch C5 – C4 nhỏ nhất phát hiện được là khoảng 4,5–6,4 điểm %. Trong khi đó, ở mức định vị, head tiếng Việt chỉ hơn head tiếng Anh khoảng 2 điểm (báo cáo tuần 4), và lỗi do dấu của B0 trên VisR-Bench chỉ 0,6 điểm F1. Kết quả âm sẽ được báo cáo kèm phân tích cơ chế.
- **Dữ liệu lệch về vài tài liệu:** 19 tài liệu, 5 tài liệu lớn nhất chiếm 64% số câu; 41% dev đến từ một tài liệu. Khoảng tin cậy báo cáo thêm bằng bootstrap theo tài liệu, và báo cáo kết quả khi bỏ lần lượt từng tài liệu lớn.
- **Một số nhóm quá nhỏ để kiểm định riêng:** câu hỏi về hình chỉ 39 câu (5 câu ở dev); nhóm độ phân giải thấp chỉ 9 câu đáp án ngắn. Các nhóm dưới 30 câu ở test chỉ báo cáo mô tả.
- **Độ tin cậy của judge chưa được xác nhận;** nhãn người dùng để đo κ được duyệt từ đề xuất của Codex nên có thể bị ảnh hưởng bởi đề xuất đó. Báo cáo kèm κ giữa judge và Codex, và mốc luật "chứa đáp án".
- **Hạn mức GPU Kaggle** khoảng 30 giờ/tuần. Mỗi nhánh crop dự kiến tốn 2–3 giờ trên dev; test của khoảng 8 nhánh tốn khoảng 40–50 giờ.

## 9. Việc tiếp theo

*Bảng 11. Việc tiếp theo*

| Việc | Sản phẩm |
|---|---|
| Nhận kết quả judge: κ của hai judge, chọn judge chính theo quy tắc, điểm judge của B0 | `results/judge/judge_selection.json`; bảng B0 theo judge |
| B1a, B1b trên dev 4 bộ | Mốc so sánh công bằng cho các nhánh crop |
| C3, C4, C5, R trên dev 4 bộ; O trên dev VisR-Bench; đo VRAM và thời gian HAVC | Kết quả RQ1–RQ3 trên dev |
| Áp quy tắc mở rộng | `results/extension_decision.json` |
| Chuẩn bị dữ liệu train, chạy thử QLoRA (VRAM, thời gian) cho C6, C7 | Cấu hình train |

## Tài liệu tham khảo

[1] J. Zhang, M. Khayatkhoei, P. Chhikara, F. Ilievski, "MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs," ICLR, 2025. arXiv:2502.17422. https://arxiv.org/abs/2502.17422

[2] J. Xie, P. Pan, X. Zhang, "Head-Aware Visual Cropping: Enhancing Fine-Grained VQA with Attention-Guided Subimage," arXiv:2601.22483, 2026. https://arxiv.org/abs/2601.22483

[3] VisR-Bench, arXiv:2508.07493 [cần bổ sung tiêu đề và tác giả từ arXiv]. Dữ liệu: https://huggingface.co/datasets/puar-playground/VisR-Bench

[4] Qwen2.5-VL Technical Report [chưa xác minh mã arXiv].

[5] TextVQA, DocVQA, ViTextVQA: bản dùng và cách lấy mẫu ghi trong `src/data_prep/sample_public.py` [cần bổ sung trích dẫn bài báo gốc của từng bộ từ arXiv].

[6] Qwen3.8-27B (bản GGUF `unsloth/Qwen3.8-27B-GGUF`), Gemma 4 12B-it (bản `google/gemma-4-12B-it-qat-q4_0-gguf`), trên Hugging Face, truy cập 18/11/2026.

[7] llama.cpp. https://github.com/ggml-org/llama.cpp

Ghi chú: mọi số liệu trong báo cáo lấy từ file trong repo: `data/qa/visr_bench_vi_v2.jsonl`, `data/qa/visr_bench_vi_v2_split.json`, `data/qa/visr_bench_vi_r2_stats.json`, `results/B0_base/`, `results/C4_havc_en/`, `results/C5_havc_vi/`.
