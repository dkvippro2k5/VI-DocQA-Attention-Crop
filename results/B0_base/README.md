# B0: model gốc trên 4 bộ dev (18/11/2026)

Qwen2.5-VL-3B-Instruct, LLM fp16, vision tower fp32, eager, `max_pixels` = 1024·28·28, prompt p2, greedy,
`max_new_tokens` = 384, T4 x2. Cấu hình: `dev_textvqa_docvqa_vitextvqa_config.json`, `dev_visr_bench_vi_config.json`.
Kết quả từng câu: `dev_<bộ>.jsonl`. EM / F1 / "chứa" lấy điểm cao nhất trên các đáp án chấp nhận; VisR-Bench chỉ nhóm `short`.

| Bộ | n | EM | Chứa đáp án | F1 | F1 bỏ dấu − F1 | Số từ dự đoán / đáp án (trung vị) | Dự đoán > 3× đáp án | s/câu |
|---|---|---|---|---|---|---|---|---|
| TextVQA | 300 | 62,7 | 78,3 | 72,8 | +0,3 | 1 / 1 | 8% | 3,0 |
| DocVQA | 300 | 82,0 | 86,0 | 88,3 | +0,0 | 2 / 2 | 3% | 3,3 |
| ViTextVQA | 300 | 25,7 | 50,7 | 51,4 | +3,3 | 4 / 3 | 17% | 2,8 |
| VisR-Bench-VI (short) | 168 | 13,7 | 50,6 | 44,6 | +0,6 | 15 / 6 | 35% | 5,6 |

Ba bộ công khai: 300 câu dev mỗi bộ (100 câu `orig` + 200 câu `ext1`, thêm ngày 18/11/2026), chạy lại cả 300 câu trong một
lượt (Kaggle, chế độ Commit; ghim `transformers==5.20.2`, torch 2.11.0, accelerate 1.15.0, qwen-vl-utils 0.0.14).
**Tái lập:** 100 câu `orig` mỗi bộ cho câu trả lời giống hệt từng ký tự lượt chạy trước (300/300), cùng phiên bản thư viện.
Không câu nào suy biến.

VisR-Bench-VI dev, toàn bộ 251 câu: EM 12,0, chứa 40,6, F1 48,6, độ phủ ý 70,6. Không câu nào suy biến;
VRAM đỉnh 7,3 GiB (GPU 0) + 3,2 GiB (GPU 1); trung bình 6,7 s/câu, lâu nhất 26,5 s (dự đoán dài).

VisR-Bench-VI dev, nhóm `short`, theo nhóm (n · EM · chứa · F1):

| Nhóm | n | EM | Chứa | F1 |
|---|---|---|---|---|
| small_detail | 29 | 27,6 | 65,5 | 45,6 |
| table | 40 | 12,5 | 22,5 | 28,3 |
| text | 99 | 10,1 | 57,6 | 51,0 |
| Độ phân giải hiệu dụng thấp / vừa / cao | 7 / 43 / 118 | 14,3 / 7,0 / 16,1 | 42,9 / 37,2 / 55,9 | 25,8 / 36,3 / 48,8 |
| Có dấu / không dấu | 127 / 41 | 12,6 / 17,1 | 53,5 / 41,5 | 48,2 / 33,6 |

Ghi chú khi đọc:
- 34/40 câu `table` ở dev thuộc một tài liệu (93906020, bảng kết quả trúng thưởng); nhóm table ở dev gần như là một tài liệu.
- Lỗi đọc thật ở bảng: "NHAN MY" → "NAM MY", "Growsure" → "Gowersur", sai số đo kích thước.
- EM thấp chủ yếu do dự đoán dài hơn đáp án (chép cả câu, cả điều khoản), không hẳn do đọc sai. "Chứa đáp án" thì
  tính đúng cả dự đoán sai mà có lẫn chuỗi đáp án (ví dụ "Số giải đặc biệt (05 chỉ vàng 9999) đã trúng: 0").
- Phần lỗi do dấu (F1 bỏ dấu − F1) trên VisR chỉ 0,6 điểm, nhỏ hơn ViTextVQA (2,4).
