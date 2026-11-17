# Giải đáp góp ý của GVHD sau báo cáo tuần 4 và đề xuất điều chỉnh

Ngày: 13/11/2026, sửa ngày 15/11/2026. Tài liệu trả lời ba góp ý của GVHD: (1) tách hai loại dữ liệu, (2) chạy baseline bằng model gốc trước khi cải tiến, (3) tự cài đặt HAVC và khả năng fine-tune. Phần cuối là kế hoạch điều chỉnh và các điểm cần GVHD chốt.

Bản sửa 15/11 đổi ba điểm so với bản 13/11:
- **Tạm gác phần truy xuất** trên tài liệu nhiều trang. ĐA2 tập trung vào bài toán hẹp và sâu: VLM đọc chi tiết nhỏ và dấu thanh trên một trang.
- **Bỏ bước khôi phục dấu** ở tiền xử lý. Chỉ giữ cách chấm bỏ dấu để phân tích.
- **Đề xuất đưa LoRA vào ĐA2** thành hai nhánh mở rộng (mục 3.3), thay cho đề xuất "không fine-tune" của bản trước.

## Tóm tắt

| Góp ý | Trả lời ngắn | Đề xuất |
|---|---|---|
| ViCrop và HAVC được đánh giá trên ảnh khác hẳn tài liệu nhiều chữ của mình; cần hai loại dữ liệu | Đúng. HAVC chưa từng được thử trên ảnh tài liệu; ViCrop chỉ thử trên DocVQA và mức tăng nhỏ. **Mình chưa kiểm chứng điều này**: 1.000 mẫu đã chạy là ảnh chữ tổng hợp dùng để chọn head, không phải để đánh giá hỏi đáp | Thiết kế 2 × 2: ngôn ngữ (Anh / Việt) × loại ảnh (ảnh cảnh có ít chữ / tài liệu nhiều chữ), mỗi ô một bộ dữ liệu |
| Chạy model gốc làm baseline, thêm tiền xử lý, rồi mới cải tiến | Đồng ý; nhánh "model gốc trên trang đúng" đã có trong kế hoạch nhưng chưa chạy | Chạy theo bậc: model gốc → tiền xử lý (tăng độ phân giải, xử lý ảnh) → crop (ViCrop, HAVC), trên cả 4 bộ dữ liệu |
| HAVC không có code nên phải tự cài; tự cài thì cũng fine-tune được | Đúng là phải tự cài (đã cài phần chọn head và phần suy luận). Nhưng tự cài **thuật toán** khác với fine-tune **trọng số**: gradient trong HAVC chỉ dùng để chấm điểm head, không cập nhật trọng số | Fine-tune làm được nếu có dữ liệu train riêng. Đề xuất thêm hai nhánh LoRA vào ĐA2: C6 (LoRA chỉ gắn vào các layer/head đã chẩn đoán) và C7 (LoRA toàn bộ). Train trên dữ liệu tiếng Việt công khai có người duyệt, không đụng tới dev/test |

## 1. Hai loại dữ liệu

### 1.1. Hai bài báo đánh giá trên dữ liệu gì

Đã đọc phần thực nghiệm của hai bài trên arXiv (bản HTML):

| | ViCrop [1] | HAVC [2] |
|---|---|---|
| Bộ đánh giá | TextVQA, V\*, POPE, DocVQA, AOKVQA, GQA, VQAv2 | AOKVQA, POPE, TextVQA, V\*, VQAv2, GQA |
| Có ảnh tài liệu nhiều chữ không | Có một bộ: DocVQA | Không |
| Model | LLaVA-1.5 và InstructBLIP (đều 7B, ảnh đầu vào độ phân giải thấp) | LLaVA-1.5 và InstructBLIP |
| Ảnh dùng để chọn head | Không chọn head | 1.000 ảnh chữ tổng hợp của Synthdog |

Hai điểm đáng chú ý:

- **Trên DocVQA, ViCrop chỉ tăng ít và độ chính xác tuyệt đối rất thấp:** LLaVA-1.5 từ 15,97 lên 19,63 (rel-att); InstructBLIP từ 9,20 lên 9,95. Bài không phân tích riêng DocVQA. Các bộ còn lại phần lớn là ảnh tự nhiên, chữ (nếu có) chỉ chiếm một vùng nhỏ.
- **Cả hai bài đều chưa thử trên model đọc được ảnh độ phân giải cao như Qwen2.5-VL.** LLaVA-1.5 thu mọi ảnh về 336 px nên crop giúp được nhiều; Qwen2.5-VL tự chia ảnh theo độ phân giải nên lợi ích của crop có thể nhỏ hơn.

Vì vậy góp ý của GVHD là có cơ sở: kết quả của hai bài chủ yếu đến từ ảnh có ít chữ, chưa nói được gì về tài liệu tiếng Việt dày chữ như bộ dữ liệu của mình.

### 1.2. Mình đã kiểm chứng điều này chưa

**Chưa.** Hai lần chạy trước đây đều không trả lời câu hỏi này:

- **1.000 mẫu của HAVC** là bước 1 (chọn head): 1.000 ảnh chữ tiếng Anh và 1.000 ảnh chữ tiếng Việt do mình tự vẽ từ Wikipedia, chỉ có chữ, không có câu hỏi. Bước này đo head nào nhìn đúng vào chữ đang đọc. Pha suy luận của HAVC (crop rồi trả lời) **chưa chạy trên bộ hỏi đáp nào**.
- **Pilot ViCrop** chỉ chạy trên 21 câu tài liệu tiếng Việt, không có bộ ảnh cảnh để so.

Tuy vậy pilot ViCrop đã cho một dấu hiệu khớp với nhận định của GVHD: với cấu hình mặc định của paper (hộp 336 px), vùng crop chỉ chiếm khoảng 3% trang và thường cắt dở câu chứa đáp án, nên kém ảnh đầy đủ (F1 0,340 so với 0,673). Phải tăng hộp lên 1024 px (khoảng 36% trang) mới vượt. Tức là tham số chọn cho ảnh tự nhiên không dùng được ngay cho trang tài liệu. Kết quả này mới dựa trên 21 câu nên chỉ là dấu hiệu.

**Vì sao cơ chế của HAVC có thể hỏng trên trang dày chữ** (giả thuyết, chưa đo):

- HAVC giữ lại các head có attention tập trung, đo bằng entropy không gian: số cụm vùng attention cao và khoảng cách giữa các cụm. Chỉ giữ head có entropy dưới 0,3; paper nói ngưỡng này lấy theo Kang và cs. và không bàn tới ảnh tài liệu.
- Trên ảnh cảnh, chữ chỉ có ở vài chỗ nên head đọc chữ cho attention tập trung, entropy thấp. Trên trang tài liệu, chữ phủ khắp trang nên attention của chính các head đọc chữ dễ rải ra nhiều dòng, tạo nhiều cụm, entropy cao. Khi đó bộ lọc có thể loại đúng các head tốt nhất, hoặc gần như không còn head nào qua lọc.
- Hướng kiểm tra rất rẻ: khi chạy HAVC, ghi lại entropy của từng head và số head qua lọc cho từng câu, rồi so phân bố giữa ảnh cảnh và ảnh tài liệu.

### 1.3. Đề xuất: thiết kế 2 × 2

Tách hai yếu tố mà GVHD nêu: loại ảnh (ít chữ hay dày chữ) và ngôn ngữ (Anh hay Việt). Mỗi ô dùng một bộ có sẵn đáp án, riêng ô tài liệu tiếng Việt dùng bộ đang tự xây:

| | Ảnh cảnh, ít chữ (giống paper) | Tài liệu, dày chữ |
|---|---|---|
| **Tiếng Anh** | TextVQA (tập validation) — cả hai paper đều dùng | DocVQA (tập validation) — ViCrop có dùng |
| **Tiếng Việt** | ViTextVQA [3] (tập test) — bộ VQA chữ trong ảnh tiếng Việt, khoảng 16 nghìn ảnh, 50 nghìn câu | VisR-Bench tiếng Việt (đang duyệt nhãn, khoảng 1.000 câu, có khung bằng chứng) |

Cách làm:

- Lấy ngẫu nhiên khoảng 500 câu mỗi bộ (seed cố định). Ba bộ công khai đã có đáp án nên không tốn công gán nhãn.
- Mọi bộ dùng chung prompt, chung model, chung script chấm. TextVQA có 10 đáp án mỗi câu thì chấm theo đáp án khớp nhất.
- Ngoài độ chính xác, đo thêm chỉ số cơ chế: entropy và số head qua lọc (HAVC), diện tích vùng crop. Với bộ tiếng Việt của mình (có khung bằng chứng), đo thêm crop có chứa bằng chứng hay không.
- Câu hỏi chính của thiết kế: mức cải thiện của ViCrop/HAVC so với model gốc thay đổi thế nào khi đổi loại ảnh và khi đổi ngôn ngữ. So sánh chính C5 vs C4 vẫn giữ trên ô tài liệu tiếng Việt.

Lưu ý:

- **Rò rỉ dữ liệu:** Qwen2.5-VL nhiều khả năng đã học tập train của TextVQA và DocVQA; Vintern có thể đã học ViTextVQA *[cần kiểm tra]*. Dùng tập validation/test giảm được phần nào nhưng điểm tuyệt đối trên các bộ này có thể bị thổi phồng. Vì vậy nên so **mức chênh trước và sau cải tiến** trong cùng một bộ, không so điểm tuyệt đối giữa các bộ.
- **Giấy phép** của ba bộ công khai cần kiểm tra trước khi tải.
- **Chi phí ước tính:** 4 bộ × 500 câu. Model gốc khoảng 1–2 giây/câu, khoảng 1 giờ; ViCrop khoảng 6 giây/câu, khoảng 3–4 giờ; HAVC chưa đo. Vừa hạn mức Kaggle 30 giờ/tuần.
- **Truy xuất (nhánh A, B, C1, C2)** chỉ có nghĩa với tài liệu nhiều trang; ảnh cảnh là một ảnh nên không có bước truy xuất. Thiết kế 2 × 2 vì vậy chạy ở chế độ "trang đúng": đưa thẳng trang chứa đáp án cho VLM. Em đề xuất **tạm gác phần truy xuất, chuyển sang khóa luận** (mục 4, mục 5).

## 2. Baseline bằng model gốc và tiền xử lý

Đồng ý với GVHD. Đề xuất chạy theo ba bậc, mỗi bậc chỉ thêm một thay đổi so với bậc trước để biết từng thay đổi đóng góp bao nhiêu:

| Bậc | Nội dung | Mục đích |
|---|---|---|
| 0. Model gốc | Qwen2.5-VL-3B và Vintern-1B, ảnh nguyên, cấu hình mặc định, prompt chung | Mức xuất phát; khoảng cách Anh – Việt và ít chữ – dày chữ khi chưa can thiệp gì |
| 1. Tiền xử lý | (a) tăng độ phân giải đầu vào (max_pixels); (b) xử lý ảnh: phóng to, làm nét, xoay thẳng trang scan | Xem các biện pháp đơn giản, không đụng tới model, đã thu hẹp khoảng cách được bao nhiêu |
| 2. Cải tiến theo attention | ViCrop, HAVC chọn head bằng tiếng Anh (C4), HAVC chọn head bằng tiếng Việt (C5), kèm đối chứng crop ngẫu nhiên | Phần cải tiến còn mang lại gì **trên nền** bậc 1 |

Về "thêm dấu tiếng Việt" trong tiền xử lý, cần làm rõ áp dụng ở đâu:

- **VLM đọc trực tiếp điểm ảnh** nên không có văn bản nào để thêm dấu ở đầu vào. Dấu chỉ có thể sai ở câu trả lời của mô hình.
- **Khôi phục dấu cho văn bản OCR** chỉ có ích ở nhánh OCR (A, B). Pilot tuần 3: ở ảnh scan trung bình, OCR đọc đúng 2/38 đáp án, và 7/38 nếu bỏ qua dấu. Nhưng em **không đưa bước này vào ĐA2**, vì hai lý do:
  - mô hình khôi phục dấu tự nó cũng sai dấu, nên thêm một nguồn lỗi chồng lên lỗi OCR và làm khó tách nguyên nhân;
  - nhánh OCR thuộc phần truy xuất, đã đề xuất tạm gác.
- **Sửa dấu cho câu trả lời của VLM:** không làm, vì có thể sửa sai một đáp án vốn đúng.
- **Thay vào đó, chấm thêm điểm khi bỏ dấu** (script đã có hàm bỏ dấu). Cách này không đưa gì thêm vào model nên không gây nhiễu. Chênh lệch giữa điểm giữ dấu và điểm bỏ dấu cho biết bao nhiêu lỗi là do dấu, phục vụ trực tiếp RQ3.

Hai lưu ý:

- Bậc 1 (a) chính là nhánh đối chứng "tăng độ phân giải" đã đề xuất trong báo cáo tuần 4, nên không phát sinh thêm việc.
- Tham số của bậc 1 (mức max_pixels, cách xử lý ảnh) phải chốt trên dev, không chọn theo kết quả test.

## 3. Tự cài HAVC và vấn đề fine-tune

### 3.1. Có phải build from scratch không

Đúng một nửa:

- **HAVC:** bài báo không nhắc tới mã nguồn và mình không tìm thấy code công khai, nên phải tự cài. Phần đã cài: chọn head bằng ảnh chữ (src/diagnosis/) và pha suy luận gồm entropy không gian, gradient, kết hợp điểm, lấy hộp crop (src/crop/havc.py). Phần suy luận chưa chạy thử.
- **ViCrop:** có mã gốc công khai (mllms_know, giấy phép MIT); bản đang dùng là bản tự viết lại theo mô tả, cần đối chiếu với mã gốc.
- **Model thì không build:** Qwen2.5-VL-3B và Vintern-1B tải nguyên trọng số đã huấn luyện từ HuggingFace. "Tự cài" ở đây là viết thuật toán bao quanh model đã có, không phải tạo hay huấn luyện model mới.

### 3.2. Gradient trong HAVC không phải fine-tune

| | Gradient trong HAVC | Fine-tune |
|---|---|---|
| Tính gradient theo | attention map của **một câu hỏi** đang trả lời | hàm mất mát trên **cả tập huấn luyện** |
| Dùng gradient để | chấm điểm mức quan trọng của từng head, rồi bỏ đi | cập nhật trọng số bằng optimizer |
| Trọng số sau đó | giữ nguyên; câu hỏi sau dùng đúng model cũ | thay đổi vĩnh viễn |
| Cần dữ liệu huấn luyện | không | có |

Nói cách khác, HAVC dùng gradient giống như một cách "hỏi" model xem head nào ảnh hưởng tới câu trả lời. Đây vẫn là can thiệp lúc suy luận, nên nằm trong phạm vi không fine-tune của ĐA2. Cái giá là phải chạy backward lúc suy luận nên tốn VRAM.

### 3.3. Đề xuất đưa LoRA vào ĐA2

Bản 13/11 đề xuất không fine-tune trong ĐA2, chủ yếu vì chưa có dữ liệu train. Từ đó đến nay đã tìm được nguồn dữ liệu (chi tiết ở `docs/Nguon_du_lieu_tieng_Viet.md`), và việc gác phần truy xuất giải phóng được thời gian. Vì vậy em đề xuất thêm hai nhánh LoRA, **nếu GVHD đồng ý đổi phạm vi**. Scope hiện ghi ĐA2 không huấn luyện, nên cần GVHD chốt trước khi làm.

**Các nhánh:**

| Nhánh | Nội dung |
|---|---|
| C6: LoRA nhắm head | Chỉ gắn LoRA vào phép chiếu attention của các layer chứa head đã chẩn đoán ở C5. Biến thể chặt hơn: chặn cập nhật ngoài các head đã chọn |
| C7: LoRA toàn bộ | Gắn LoRA vào mọi layer attention, cùng dữ liệu, cùng số bước train. Đây là đối chứng để biết việc nhắm head có tác dụng không |

**Câu hỏi nghiên cứu:** LoRA nhắm vào các head đã chẩn đoán, train trên dữ liệu VQA tiếng Việt giàu chữ nhưng không phải tài liệu, có chuyển được sang hỏi đáp tài liệu tiếng Việt không, và có hơn can thiệp training-free C5 không?

**Dữ liệu train** (chỉ dùng phần train, không trùng tài liệu với dev/test):

| Bộ | Cỡ (phần train) | Nhãn | Tình trạng |
|---|---|---|---|
| ViInfographicVQA [4] | 12.521 câu single-image; lọc câu trích từ ảnh, ưu tiên loại Text/Table còn khoảng 4–6 nghìn | Gemini sinh, người duyệt | Đã tải |
| ViTextVQA [3] | Phần train của 50 nghìn câu | Người gán *[cần kiểm tra quy trình]* | Đã được duyệt truy cập |
| MTVQA, phần tiếng Việt [5] | 4.011 câu | Người bản ngữ gán | Công khai |

Tất cả đều có người viết hoặc người duyệt. Không dùng nhãn LLM chưa duyệt.

**Tài nguyên:**
- QLoRA cho Qwen2.5-VL-3B vừa T4 16 GB khi giới hạn max_pixels.
- Lúc train không cần lấy attention weights nên không bắt buộc chế độ eager.
- Ước tính vài giờ cho một lượt khoảng 10 nghìn mẫu *[chưa đo, cần chạy thử]*. Hạn mức Kaggle khoảng 30 giờ/tuần, đủ cho 3–4 cấu hình.

**Giữ các quy tắc hiện có:**
- Siêu tham số LoRA (rank, learning rate, số bước) chọn trên dev; test chạy một lần.
- So sánh chính C5 vs C4 không đổi. C6, C7 so với C5 là so sánh phụ, có hiệu chỉnh Holm.
- Báo cáo cả kết quả âm: nếu LoRA thua C5 hoặc không chuyển được sang tài liệu, đó vẫn là kết luận.

**Rủi ro chính: lệch miền.** Dữ liệu train là infographic, ảnh cảnh, không phải trang tài liệu dày chữ. Thiết kế 2 × 2 giúp đo đúng điều này: ô ảnh cảnh tiếng Việt (ViTextVQA) gần miền train, ô tài liệu tiếng Việt là chuyển miền. Song song, nhóm đang xin dữ liệu cùng miền: phần tiếng Việt của SEA-Vision và Viet-Doc-VQA-II (đang chờ duyệt; nếu được thì kiểm 100 mẫu trước khi dùng).

**Phần training-free vẫn giữ nguyên.** Siêu tham số của ViCrop/HAVC (ngưỡng entropy, K, α, τ, layer, cỡ hộp crop) vẫn chỉnh trên dev bằng tìm kiếm lưới. Đây chính là chỗ góp ý 1 nhắm tới: ngưỡng entropy 0,3 được chọn cho ảnh tự nhiên, có thể cần ngưỡng khác cho trang dày chữ. Mọi thay đổi so với paper được ghi rõ.

## 4. Kế hoạch điều chỉnh

| Bước | Công việc | Sản phẩm |
|---|---|---|
| 1 | Kiểm tra giấy phép; tải và lấy mẫu 500 câu mỗi bộ TextVQA, DocVQA, ViTextVQA | 3 file câu hỏi cùng định dạng với bộ tiếng Việt |
| 2 | Bậc 0: Qwen2.5-VL-3B và Vintern-1B gốc trên 4 bộ | Bảng 2 × 2 của model gốc |
| 3 | Bậc 1: tăng độ phân giải, xử lý ảnh; thêm điểm chấm khi bỏ dấu | Mức thu hẹp khoảng cách nhờ tiền xử lý |
| 4 | Bậc 2: ViCrop, HAVC-EN (C4), HAVC-VI (C5), crop ngẫu nhiên trên 4 bộ; ghi entropy và số head qua lọc | Mức cải thiện theo loại ảnh và ngôn ngữ, kèm phân tích cơ chế |
| 5 | Nếu GVHD đồng ý: chuẩn bị dữ liệu train, chạy thử QLoRA đo VRAM và thời gian, rồi train C6 và C7 | Hai nhánh LoRA chạy được trên dev |
| 6 | Chỉnh siêu tham số trên dev (training-free và LoRA); chốt cấu hình; chạy test một lần | Kết quả cuối có kiểm định thống kê |

Phần truy xuất (A, B, C1, C2) trên tài liệu nhiều trang chuyển sang khóa luận.

Song song: duyệt xong nhãn vòng 2 của bộ tiếng Việt (bước 2–4 có thể chạy thử trên 109 câu đã duyệt, chạy lại khi đủ nhãn) và chờ phản hồi xin dữ liệu.

## 5. Các điểm cần GVHD chốt

1. **Thiết kế 2 × 2** với bốn bộ ở mục 1.3, khoảng 500 câu mỗi bộ, chạy không huấn luyện. Em thấy phù hợp; thầy/cô có đồng ý không?
2. **Gác phần truy xuất** (A, B, C1, C2) sang khóa luận để ĐA2 tập trung vào bài toán đọc trên một trang. Thầy/cô có đồng ý không?
3. **"Thêm dấu tiếng Việt":** em đề xuất không khôi phục dấu (lo thêm nhiễu), chỉ chấm thêm điểm khi bỏ dấu để đo lỗi do dấu (mục 2). Nếu thầy/cô có ý khác về bước tiền xử lý này, em xin được hướng dẫn thêm.
4. **Đưa LoRA vào ĐA2** thành nhánh C6/C7 như mục 3.3, tức là đổi quy định "ĐA2 không huấn luyện" trong scope. Thầy/cô có đồng ý không?
5. **Dữ liệu train** là các bộ tiếng Việt công khai có người viết hoặc người duyệt (mục 3.3); tập tiếng Việt tự xây chỉ dùng để đánh giá. Thầy/cô có đồng ý không?

## Tài liệu tham khảo

[1] J. Zhang, M. Khayatkhoei, P. Chhikara, F. Ilievski, "MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs," ICLR, 2025. arXiv:2502.17422. https://arxiv.org/abs/2502.17422 — đã đọc phần thực nghiệm (bản HTML).

[2] J. Xie, P. Pan, X. Zhang, "Head-Aware Visual Cropping: Enhancing Fine-Grained VQA with Attention-Guided Subimage," arXiv:2601.22483, 2026. https://arxiv.org/abs/2601.22483 — đã đọc phần phương pháp và thực nghiệm (bản HTML).

[3] ViTextVQA, arXiv:2404.10652. https://arxiv.org/abs/2404.10652 ; dữ liệu: https://huggingface.co/datasets/minhquan6203/ViTextVQA — *[mới xem trang tóm tắt; cần kiểm tra tác giả và cách chia tập]*. Trang Hugging Face ghi giấy phép CC BY-NC 3.0.

[4] T.-T. Van-Dinh, H.-D. Tran, T.-B. Duong, M.-H. Pham, B.-N. Le-Nguyen, Q.-T. Nguyen, "ViInfographicVQA: A Benchmark for Single and Multi-image Visual Question Answering on Vietnamese Infographics," arXiv:2512.12424, 2025. https://arxiv.org/abs/2512.12424 ; dữ liệu: https://huggingface.co/datasets/duytranus/ViInfographicVQA — đã đọc phần dữ liệu; tên tác giả lấy từ trang đầu PDF.

[5] MTVQA, dữ liệu: https://huggingface.co/datasets/ByteDance/MTVQA (CC BY-NC 4.0) — *[mới xem trang dữ liệu; chưa đọc paper]*.
