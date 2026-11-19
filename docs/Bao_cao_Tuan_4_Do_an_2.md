# BÁO CÁO TUẦN 4 ĐỒ ÁN 2

**Đề tài:** Vision-Language Model trong hỏi đáp tài liệu tiếng Việt
**Sinh viên:** Dương Văn Kiên
**MSSV:** 20233857
**GVHD:** Dr. Đỗ Thị Ngọc Diệp
**Nội dung:** Mở rộng bộ dữ liệu, chẩn đoán head cho HAVC, chốt giao thức đánh giá; nối tiếp báo cáo tuần 3

---

## 1. Tóm tắt

Tuần này tập trung vào dữ liệu và bước chẩn đoán head của HAVC. Kết quả chính là rủi ro lớn nhất của scope (VLM không định vị được chữ tiếng Việt) đã được loại bỏ; ngược lại, khác biệt giữa head chọn bằng chữ tiếng Việt và bằng chữ tiếng Anh nhỏ hơn kỳ vọng.

- **Dữ liệu:** mở rộng ra toàn bộ phần tiếng Việt tải được của VisR-Bench: 23 tài liệu, 1.045 câu hỏi sau khi loại 2 tài liệu nhạy cảm. Gemini Flash đã gán đáp án cho cả 1.045 câu. 109 câu vòng 1 đã duyệt tay xong (giữ nguyên 91, sửa 18); 936 câu vòng 2 đang được duyệt tay toàn bộ kèm khung bằng chứng.
- **Chẩn đoán head (bước 1 của HAVC):** trên 1.000 ảnh chữ mỗi ngôn ngữ, Qwen2.5-VL-3B đọc đúng 99,2–99,7% token; head tốt nhất định vị đúng chữ 86–91% số token, so với khoảng 1% nếu đoán ngẫu nhiên.
- **Head tiếng Việt và tiếng Anh:** trên nửa ảnh tiếng Việt giữ lại, head chọn bằng chữ tiếng Việt định vị tốt hơn head chọn bằng chữ tiếng Anh 2,1 điểm % (2,5 điểm ở từ có dấu), có ý nghĩa thống kê. Mức chênh này nhỏ và hai bộ head gần giống nhau (Spearman 0,988), nên chênh lệch C5–C4 trên tác vụ hỏi đáp có thể nhỏ.
- **Giao thức đánh giá:** chốt prompt sinh đáp án chung cho mọi nhánh và cách chấm theo dạng đáp án (ngắn / dài / nhiều ý) trước khi chạy các nhánh.
- **Chưa làm:** nhánh A, B, C1, C2; chưa chạy thử pha suy luận của HAVC nên chưa có số VRAM.


## 2. Dữ liệu

### 2.1. Mở rộng bộ dữ liệu

Bộ pilot tuần 3 chỉ có 5 tài liệu. Tuần này tải toàn bộ phần tiếng Việt của VisR-Bench [4] (cố định phiên bản trên HuggingFace để tái lập) và tải thêm PDF gốc từ đường dẫn của từng tài liệu, vì lớp text PDF cho tọa độ từ chính xác hơn OCR (Bảng 1).

*Bảng 1. Quy mô bộ dữ liệu sau khi mở rộng*

| Hạng mục | Số lượng | Ghi chú |
|---|---|---|
| Tài liệu tiếng Việt đã tải | 25 / 27 | 2 tài liệu dài hơn 60 trang bị bỏ qua khi tải |
| Tài liệu bị loại | 2 | Bài báo An Bằng (chính trị, đã loại từ tuần 3) và bản tin của một tổ chức tôn giáo (2 trang, 6 câu) |
| Tài liệu dùng | 23 | Số câu mỗi tài liệu từ 5 đến 153, lệch nhiều |
| Câu hỏi tiếng Việt | 1.045 | 109 câu vòng 1 và 936 câu vòng 2 |
| PDF gốc tải được | 15 / 25 | 10 đường dẫn hỏng; 2 PDF gần như không có lớp text (bản scan) |
| Trang của câu vòng 2 cần OCR | 276 / 322 | Không có lớp text PDF dùng được; dùng EasyOCR |

Danh sách tài liệu bị loại được lưu trong data/qa/excluded_documents.json để mọi bước sau tự bỏ qua; tài liệu mới được kiểm tra nội dung trước khi gán nhãn.

### 2.2. Gán đáp án vòng 2

936 câu còn lại được gán bằng bốn worker Gemini Flash trong Antigravity, cùng model và cùng prompt (bản v3). So với v2, v3 bỏ trần 25 từ cho câu hỏi nhiều ý và yêu cầu chép đủ các ý, ngăn bằng dấu chấm phẩy, vì ở vòng 1 nhiều câu liệt kê bị cắt còn một ý. Mỗi worker làm thêm 12 câu hiệu chuẩn chung. Một script kiểm tra tự động chạy sau mỗi batch: loại dòng sai định dạng, gắn cờ cảnh báo và soạn sẵn tin nhắc sửa cho từng worker.

Kết quả: 936/936 câu hợp lệ; 45 câu được đánh là không trả lời được từ trang; 63 câu thiếu trường loại bằng chứng (trường phụ, không ảnh hưởng chấm điểm). Kiểm tra thủ công hai worker có tốc độ bất thường cho thấy worker nhanh nhất không có lỗi hệ thống, còn worker chậm nhất hơi thận trọng quá mức (có câu trả lời được nhưng bị đánh là không trả lời được). Các câu đánh không trả lời được đều được xếp duyệt trước.

### 2.3. Duyệt tay và khung bằng chứng

Việc duyệt dùng một trang HTML chạy trên máy: hiển thị ảnh trang, đáp án Gemini, khung gợi ý dò tự động từ tọa độ từ (lớp text PDF, hoặc EasyOCR nếu không có PDF), cho phép vẽ, xóa khung và phóng/thu ảnh, và xếp các câu đáng ngờ lên đầu. Mỗi câu được giữ nguyên, sửa hoặc bỏ (kèm lý do). Câu có nhiều khung ghi rõ các khung là các phần của đáp án hay là các vị trí thay thế của cùng một đáp án.

*Bảng 2. Kết quả duyệt vòng 1 (109 câu)*

| Chỉ số | Kết quả |
|---|---|
| Đáp án Gemini giữ nguyên / phải sửa | 91 / 18 (tỉ lệ chấp nhận 83,5%) |
| Khung gợi ý dùng nguyên / phải chỉnh / phải vẽ tay | 43 / 9 / 26 (trên 78 câu có ghi nguồn khung) |
| Khung gợi ý dùng nguyên, trang có text PDF | 41 / 70 (59%) |
| Khung gợi ý dùng nguyên, trang phải OCR | 2 / 8 |
| Thời gian duyệt mỗi câu | trung vị 42 giây, trung bình 85 giây |

Gợi ý khung hoạt động tạm được trên trang có text PDF nhưng gần như không dùng được trên trang scan, do EasyOCR sai dấu, rớt từ và đọc lẫn các cột. Vì 86% trang của vòng 2 là trang scan, phần lớn khung vòng 2 phải vẽ tay. Đã thử một cách thay thế: cho worker chỉ ra số dòng chứa đáp án trong văn bản OCR đã đánh số để dựng khung tự động. Cách này đã được bỏ trước khi chạy thật, vì cần thêm một lượt gán cho cả 936 câu trong khi văn bản OCR của trang scan quá nhiễu để worker chọn dòng đáng tin cậy. Quyết định cuối là duyệt tay 100% vòng 2 (ước tính 11–22 giờ), không dùng nhãn chưa duyệt.

### 2.4. Dạng đáp án

Đáp án được tự xếp thành ba dạng theo quy tắc cố định: nhiều ý (có từ 2 ý ngăn bằng dấu chấm phẩy), dài (một ý, hơn 25 từ) và ngắn (còn lại). Quy tắc được chốt trước khi có kết quả của nhánh nào và dùng chung cho bước gộp nhãn và bước chấm điểm (Bảng 3).

*Bảng 3. Phân bố dạng đáp án*

| Dạng | Vòng 1 (đã duyệt) | Vòng 2 (theo nhãn Gemini, chưa duyệt) |
|---|---|---|
| Ngắn | 98 | 598 |
| Nhiều ý | 6 | 241 |
| Dài | 5 | 52 |

Vòng 2 có tỉ lệ đáp án nhiều ý cao hơn hẳn vòng 1 (27% so với 6%). Đáp án nhiều ý có trung vị 3 ý và 41 từ; 10% dài hơn 139 từ.

## 3. Chẩn đoán head cho HAVC (C4, C5)

HAVC [2] chọn các attention head "biết đọc chữ" bằng một tác vụ OCR trên ảnh chữ tiếng Anh, rồi dùng chúng để chọn vùng crop. Đề xuất của ĐA2 (C5) là chọn head bằng ảnh chữ tiếng Việt có dấu. Bước này kiểm tra hai điều: mô hình có head nào định vị được chữ tiếng Việt không, và chọn head bằng tiếng Việt có khác chọn bằng tiếng Anh không.

### 3.1. Thiết lập

- Ảnh: 1.000 ảnh chữ tiếng Anh và 1.000 ảnh chữ tiếng Việt, tự vẽ từ đoạn văn Wikipedia (bỏ bài có chủ đề nhạy cảm). Ảnh thứ i của hai bộ dùng chung bố cục (font, cỡ chữ, nền, kích thước, nhiễu), chỉ khác văn bản; 8 font mã nguồn mở có đủ dấu tiếng Việt. Ảnh tiếng Việt có trung bình 55 từ, 80% là từ có dấu; ảnh tiếng Anh có 43 từ.
- Mô hình: Qwen2.5-VL-3B-Instruct [5], fp16, attention eager, max_pixels = 1024×28×28, chạy trên Kaggle T4. Đưa văn bản đúng làm câu trả lời (teacher forcing), mỗi ảnh một lần forward: 0,6 giây mỗi ảnh, VRAM đỉnh 4,6 GB.
- Điểm head theo paper: với mỗi token mô hình đọc đúng, một head được tính là trúng nếu đỉnh attention rơi vào vùng chữ của từ chứa token đó; điểm được chuẩn hóa min-max trên toàn bộ 36 layer × 16 head, giữ head có điểm trên 0,5 và xếp hạng để lấy K = 8.
- Sai khác so với paper: paper dùng ảnh Synthdog nhưng không kèm khung chữ, nên ở đây tự vẽ ảnh để có khung từng từ; cách định nghĩa vùng chữ, tập token đầu vào được xét và lưới token ảnh của Qwen2.5-VL (ô 28×28 px, không vuông) là tự chọn và được ghi trong code.

### 3.2. Kết quả

*Bảng 4. Kết quả chẩn đoán trên hai bộ ảnh*

| Chỉ số | Chữ tiếng Anh | Chữ tiếng Việt |
|---|---|---|
| Số token được xét | 57.626 | 76.444 |
| Tỉ lệ token đọc đúng | 99,7% | 99,2% |
| Tỉ lệ trúng của head tốt nhất | 0,906 (layer 32, head 7) | 0,857 (layer 32, head 7) |
| Tỉ lệ trúng trung bình mọi head | 0,125 | 0,116 |
| Tỉ lệ trúng nếu đoán ngẫu nhiên | 0,014 | 0,011 |
| Số head có điểm trên ngưỡng 0,5 | 35 | 44 |

![Hình 1](img/bao_cao_tuan_4_hinh1.png)

*Hình 1. Tỉ lệ trúng của head tốt nhất ở từng layer; đường nét đứt là mức đoán ngẫu nhiên*

Các head định vị chữ tốt tập trung ở cùng các layer cho cả hai ngôn ngữ (5, 7, 20, 22 và 25–32), và ở phần lớn các layer, head tốt nhất định vị chữ tiếng Anh tốt hơn chữ tiếng Việt một chút (Hình 1). Hai bộ head khá giống nhau: tương quan hạng Spearman của điểm head là 0,988; trong 8 head tốt nhất có 5 head chung (Jaccard 0,455); trong các head trên ngưỡng, Jaccard là 0,717.

Để so sánh công bằng, chọn 8 head trên nửa ảnh chẵn của mỗi bộ, rồi đo khả năng định vị trên nửa ảnh lẻ của bộ tiếng Việt (không dùng để chọn head). Khoảng tin cậy lấy bằng paired bootstrap theo ảnh (5.000 lần, 498 ảnh), hiệu chỉnh Holm cho 4 kiểm định (Bảng 5).

*Bảng 5. Tỉ lệ trúng trên nửa ảnh tiếng Việt giữ lại*

| Nhóm từ | Head chọn bằng tiếng Việt | Head chọn bằng tiếng Anh | 8 head ngẫu nhiên | Chênh lệch (KTC 95%) |
|---|---|---|---|---|
| Tất cả | 0,797 | 0,776 | 0,114 | +2,1 điểm (1,9–2,3) |
| Từ có dấu | 0,811 | 0,786 | 0,107 | +2,5 điểm (2,3–2,7) |
| Từ không dấu | 0,758 | 0,746 | 0,133 | +1,2 điểm (0,9–1,5) |

Cả ba chênh lệch đều có p < 0,001 sau hiệu chỉnh Holm. Lợi thế của head tiếng Việt ở từ có dấu lớn hơn ở từ không dấu 1,3 điểm (KTC 95%: 0,9–1,6), phù hợp với giả thuyết của RQ3 ở mức định vị.

Cách hiểu kết quả:

- Rủi ro "VLM 1–3B định vị chữ tiếng Việt kém ở mọi head" (scope mục 12) đã được loại bỏ: head tốt nhất định vị đúng 86% token tiếng Việt.
- Chênh lệch giữa hai cách chọn head có thật nhưng nhỏ (khoảng 2 điểm %), đo trên ảnh tổng hợp và ở mức định vị từng token. Chưa biết chênh lệch này có chuyển thành chênh lệch đáp án hay không; nếu có, nhiều khả năng cũng nhỏ. Với cỡ test dự kiến khoảng 500 câu, chỉ phát hiện được chênh lệch khoảng 5 điểm %, nên kết quả C5 vs C4 có thể không có ý nghĩa thống kê. Kết quả âm như vậy vẫn sẽ được báo cáo.

### 3.3. Pha suy luận

Code pha suy luận của HAVC đã viết xong: lọc head bằng entropy không gian, độ nhạy gradient, kết hợp điểm và lấy hộp crop theo ViCrop. Paper không cho hai hệ số của entropy không gian nên tạm chọn λc = 0,1 và λd = 0,5, sẽ ghi là sai khác. Pha này cần backward lúc suy luận nên chưa có số VRAM; sẽ chạy thử tuần tới.

## 4. Giao thức đánh giá

### 4.1. Prompt sinh đáp án chung

Prompt cũ ("Trả lời ngắn gọn bằng một cụm từ hoặc một câu.", tối đa 64 token) bất lợi cho 241 câu nhiều ý: mô hình chỉ nêu một ý và đáp án dài bị cắt cụt. Prompt mới dùng cho mọi câu và mọi nhánh (nhánh A dùng cùng câu chỉ dẫn, thay ảnh bằng văn bản):

"Trả lời câu hỏi dựa vào tài liệu. Chép nguyên văn từ tài liệu, ngắn nhất mà vẫn đủ ý. Nếu câu hỏi yêu cầu nhiều ý (liệt kê, các bước, các điều kiện, các thành phần), nêu đủ mọi ý, mỗi ý ngắn gọn, ngăn cách bằng dấu chấm phẩy. Chỉ ghi đáp án, không giải thích."

Tối đa 384 token, giải mã tham lam. Không chọn prompt riêng theo loại câu hỏi, vì như vậy là dùng thông tin nhãn lúc suy luận. Kết quả C3 tuần 3 dùng prompt cũ nên sẽ chạy lại trên bộ dev mới.

### 4.2. Chấm theo dạng đáp án

Kết quả được báo cáo tách theo ba dạng đáp án ở mục 2.4. Câu nhiều ý được chấm thêm độ phủ theo ý: trung bình theo từng ý của tỉ lệ âm tiết của ý đó có trong câu trả lời, vì F1 trên cả chuỗi dài phạt nặng mọi nhánh như nhau và che mất khác biệt. Phân tích chính của RQ2 (crop) dựa trên nhóm câu ngắn, vì đáp án dài trải trên vùng lớn mà crop không nhắm tới.

### 4.3. Đề xuất hai nhánh đối chứng

Crop đưa thêm ảnh vào mô hình, tức là cho mô hình thêm điểm ảnh ở vùng chữ. Để tách tác dụng của việc chọn đúng vùng khỏi tác dụng của việc có thêm token ảnh, em đề xuất thêm hai nhánh:

- **Tăng độ phân giải:** nhánh C tốt nhất với max_pixels tăng sao cho tổng token ảnh xấp xỉ ảnh gốc cộng ảnh crop của cấu hình C3–C5 đã chốt. Hiện ảnh trang khoảng 1800×2339 px bị thu nhỏ hơn 5 lần xuống 1.024 token.
- **Crop ngẫu nhiên:** crop cùng kích thước tại vị trí ngẫu nhiên (seed cố định).

Nếu pha suy luận HAVC vượt VRAM của T4, thứ tự dự phòng là: chạy trên 2×T4; bỏ thành phần gradient (chỉ dùng entropy), vẫn giữ được so sánh C5 vs C4 vì hai nhánh chỉ khác bộ head; sau cùng mới chuyển sang FOCUS [6].

## 5. Tình trạng các nhánh

*Bảng 6. Tình trạng các nhánh thí nghiệm*

| Nhánh | Tình trạng |
|---|---|
| A (OCR + LLM), B (RAG văn bản + VLM) | Chưa bắt đầu |
| C1 (ColVintern + Vintern-1B), C2 (ColQwen + Qwen2.5-VL-3B) | Chưa bắt đầu |
| C3 (ViCrop) | Pilot trên 21 câu dev cũ với trang đúng (chưa qua truy xuất), prompt cũ; sẽ chạy lại |
| C4, C5 (HAVC) | Đã chọn xong head; code suy luận đã viết, chưa chạy thử |

Theo kế hoạch, các nhánh A, B, C1, C2 làm trong tuần 5–7, nên chưa trễ, nhưng sẽ phải làm song song với việc duyệt nhãn vòng 2.

## 6. Hạn chế và rủi ro

- **Quy mô dữ liệu:** khoảng 1.000 câu từ 23 tài liệu, ít hơn mức "vài nghìn mẫu" mong muốn. Số câu mỗi tài liệu lệch mạnh (5 đến 153), nên khi chia dev/test theo tài liệu, test khó vừa đủ lớn vừa đa dạng.
- **Chênh lệch C5–C4 nhỏ:** ở mức định vị chỉ khoảng 2 điểm %; khó có ý nghĩa thống kê trên tác vụ hỏi đáp.
- **Chất lượng nhãn:** đáp án do Gemini gán và một người duyệt; chưa đo độ đồng thuận giữa người duyệt.
- **Thiên lệch nguồn nhãn:** nếu nhánh A hoặc B dùng Gemini làm LLM, nhãn do Gemini sinh có thể thiên vị nhánh đó.
- **Trang scan chiếm đa số:** OCR kém làm khung gợi ý gần như vô dụng và ảnh hưởng mạnh tới nhánh A, B (đây cũng là điều đồ án muốn đo).
- **Rò rỉ dữ liệu:** chưa kiểm tra liệu VisR-Bench có nằm trong dữ liệu huấn luyện của Qwen2.5-VL hay không.

*Bảng 7. Rủi ro và phương án*

| Rủi ro | Phương án dự phòng |
|---|---|
| Test quá nhỏ để thấy khác biệt C5–C4 | Báo cáo khoảng tin cậy và kích thước hiệu ứng thay vì chỉ p; bổ sung phân tích định vị trên bbox đáp án thật |
| Pha suy luận HAVC vượt VRAM T4 | Thứ tự dự phòng ở mục 4.3 |
| Duyệt nhãn vòng 2 kéo dài | Viết code các nhánh song song, chạy thử trên 109 câu v1; chạy lại khi đủ nhãn |
| Thiên lệch do nhãn Gemini | Dùng LLM khác Gemini cho nhánh A, B, hoặc nêu rõ trong báo cáo |

## 7. Kế hoạch tuần tới

*Bảng 8. Công việc tuần tới*

| Công việc | Sản phẩm |
|---|---|
| Duyệt xong 936 câu vòng 2, gộp với vòng 1 | Bộ nhãn khoảng 1.000 câu có khung bằng chứng |
| Chia dev/test theo tài liệu; gán loại câu hỏi theo quy tắc | Danh sách tài liệu dev/test, bảng phân bố loại câu hỏi |
| Script đánh giá dùng chung kèm kiểm định McNemar, paired bootstrap, Holm | Script đánh giá |
| Nhánh mốc: VLM trả lời trên trang đúng, không crop | Kết quả mốc cho C3–C5 |
| Bắt đầu nhánh C2 (ColQwen + Qwen2.5-VL-3B), đo Recall@K của truy xuất | Kết quả truy xuất trên dev |
| Chạy thử pha suy luận HAVC | Số đo VRAM và thời gian mỗi câu |

## Tài liệu tham khảo

[1] J. Zhang, M. Khayatkhoei, P. Chhikara, F. Ilievski, "MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs," ICLR, 2025. arXiv:2502.17422. https://arxiv.org/abs/2502.17422

[2] J. Xie, P. Pan, X. Zhang, "Head-Aware Visual Cropping: Enhancing Fine-Grained VQA with Attention-Guided Subimage," arXiv:2601.22483, 2026. https://arxiv.org/abs/2601.22483

[3] Mã nguồn ViCrop (mllms_know), MIT. https://github.com/saccharomycetes/mllms_know

[4] VisR-Bench, arXiv:2508.07493 [cần bổ sung tiêu đề và tác giả từ arXiv]. Dữ liệu: https://huggingface.co/datasets/puar-playground/VisR-Bench

[5] Qwen2.5-VL Technical Report [chưa xác minh mã arXiv].

[6] "FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering," arXiv:2506.21710 [cần bổ sung tác giả từ arXiv]. https://arxiv.org/abs/2506.21710

[7] Wikimedia Foundation, bộ dữ liệu wikimedia/wikipedia (bản 20231101). https://huggingface.co/datasets/wikimedia/wikipedia

Ghi chú: mô tả phương pháp HAVC ở mục 3 dựa trên phần phương pháp của bài báo [2] (đã đọc qua bản HTML trên arXiv). Mọi số liệu trong báo cáo lấy từ file kết quả trong repo (results/C4_havc_en, results/C5_havc_vi, data/qa).
