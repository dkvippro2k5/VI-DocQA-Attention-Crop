# Nguồn dữ liệu hỏi đáp ảnh/tài liệu tiếng Việt (khảo sát 13/11/2026)

Mục đích: tìm mọi nguồn dữ liệu có thể dùng cho (a) đánh giá bổ sung ở ĐA2 (thiết kế 2 × 2) và (b) dữ liệu train cho LoRA ở khóa luận. Bổ sung cho khảo sát 27/10/2026 (V-Doc, ViInfographicVQA, VisR-Bench, MTVQA).

Mức kiểm chứng: các con số lấy từ trang paper (arXiv/ACL Anthology) hoặc trang Hugging Face của bộ dữ liệu, đọc ngày 13/11/2026. Mục ghi *[cần kiểm chứng]* là mới thấy qua kết quả tìm kiếm hoặc trang tóm tắt, chưa đọc nguồn gốc.

## 1. Tiêu chí "hợp lệ"

1. **Nhãn do người viết hoặc người duyệt.** Bộ do LLM sinh mà không có người duyệt thì loại (theo quyết định ngày 13/11: không dùng nhãn LLM chưa duyệt).
2. **Giấy phép cho phép nghiên cứu học thuật.** CC BY-NC dùng được cho đồ án/khóa luận, không dùng thương mại.
3. **Không trùng tài liệu với dev/test của mình.**
4. **Kiểm tra rò rỉ:** bộ đã nằm trong dữ liệu train của Vintern/Qwen2.5-VL thì không dùng làm test, và lợi ích khi train lại cũng thấp.

Lưu ý về tiêu chí 1: bộ test của mình cũng làm theo quy trình "LLM viết nháp, người duyệt toàn bộ". Nếu chấp nhận quy trình này cho test thì cũng phải chấp nhận cho train, với điều kiện việc duyệt là thật. Vì vậy với mọi bộ nhóm B dưới đây, mình đề xuất **tự lấy ngẫu nhiên 100 mẫu kiểm tra** và ghi tỉ lệ lỗi trước khi dùng.

## 2. Nhóm A: nhãn do người viết

| Bộ | Loại ảnh | Quy mô | Gán nhãn | Bbox | Giấy phép, truy cập | Dùng cho |
|---|---|---|---|---|---|---|
| **ViTextVQA** (arXiv 2404.10652) | Ảnh cảnh có chữ | 16.762 ảnh, 50.342 câu | Người gán *[cần kiểm chứng chi tiết quy trình]* | Không rõ | CC BY-NC 3.0; HF có cổng đăng ký (`minhquan6203/ViTextVQA`) | Ô "tiếng Việt, ảnh cảnh" của thiết kế 2 × 2; train |
| **MTVQA, phần tiếng Việt** (ByteDance) | Ảnh cảnh và tài liệu | Train 1.139 ảnh / 4.011 câu; test 250 ảnh / 884 câu | Người bản ngữ gán theo quy tắc | Không | CC BY-NC 4.0; HF `ByteDance/MTVQA` | Train (nhỏ); test phụ, cần kiểm tra Qwen2.5-VL đã dùng làm benchmark hay làm dữ liệu train |
| **OpenViVQA** (UIT, 2023) | Ảnh đời sống Việt Nam, khoảng một nửa có chữ | 11.199 ảnh, 37.914 câu | Người viết | Không | HF `uitnlp/OpenViVQA-dataset` *[cần kiểm chứng giấy phép]* | Train; đáp án dạng diễn giải, khó chấm EM |
| **EVJVQA** (VLSP 2022) | Ảnh đời sống Việt Nam, 3 ngôn ngữ | Khoảng 4,9 nghìn ảnh, 33,8 nghìn câu (cả 3 ngôn ngữ) | Người viết | Không | Qua VLSP *[cần kiểm chứng]* | Train phụ |
| **ViSignVQA** (arXiv 2512.22218) | Biển hiệu | 10.762 ảnh, 25.573 câu, chia 7:1:2 | Câu hỏi lấy lại từ OpenViVQA và ViTextVQA, 2 người bản ngữ lọc và sửa | Không công bố rõ | Chưa thấy giấy phép và link | **Trùng với ViTextVQA và OpenViVQA**, không dùng cùng lúc |

## 3. Nhóm B: LLM sinh, người duyệt

| Bộ | Loại ảnh | Quy mô | Gán nhãn | Bbox | Giấy phép, truy cập | Dùng cho |
|---|---|---|---|---|---|---|
| **ViInfographicVQA** (arXiv 2512.12424) | Infographic | Train 18.399 câu / 6.096 ảnh; test 2.010 câu / 651 ảnh | Gemini 2.0 Flash sinh, người duyệt và sửa | Paper có bước trích vùng nhưng **không công bố tọa độ** | HF `duytranus/ViInfographicVQA` (không khóa); README không ghi giấy phép | Train chính (gần tài liệu nhất trong nhóm có người duyệt). **Đã tải 13/11** vào `data/raw/viinfographicvqa` (commit `db06072`); số câu, số ảnh khớp paper |
| **ChartTableViVQA** (AACL 2026) | Biểu đồ, bảng từ nền tảng giáo dục | Bảng 1: train 9.121 / dev 1.951 / test 1.931 câu trên 868 ảnh. Phần tóm tắt ghi 13.003 bộ ba, chưa khớp | GPT-4o sinh, người Việt duyệt | Không | GitHub `sonlam1102/charttablevivqa`; giấy phép chưa ghi | Train cho câu hỏi dạng bảng |
| **SEA-Vision** (CVPR 2026, arXiv 2603.15409) | Tài liệu (9 loại) và ảnh cảnh, 11 ngôn ngữ Đông Nam Á | Tổng 15.234 trang phân tích tài liệu, 7.496 câu VQA; chưa rõ phần tiếng Việt | MLLM gán, người bản ngữ duyệt nhẹ | Nhãn phân cấp trang/khối/dòng (có thể có tọa độ) | Chưa thấy link tải | **Nên xin tác giả**: là bộ duy nhất có trang tài liệu tiếng Việt kèm nhãn mức dòng |
| **ViExam** (arXiv 2508.13680) | Đề thi có hình | 2.548 câu, 7 lĩnh vực | 3 người bản ngữ duyệt | Không | *[cần kiểm chứng]* | Chỉ để đánh giá suy luận; trắc nghiệm, lệch đề tài |
| **VLSP 2025 MLQA-TSR** | Luật giao thông kèm biển báo | *[cần kiểm chứng]* | *[cần kiểm chứng]* | Không | Qua VLSP | Lệch đề tài (trắc nghiệm, truy xuất luật) |

## 4. Nhóm C: loại theo tiêu chí 1 (LLM sinh, không người duyệt)

| Bộ | Lý do loại |
|---|---|
| Viet-Doc-VQA, Viet-Doc-VQA-II (64.765 trang sách giáo khoa, 388.277 mục), Viet-Doc-VQA-verIII, Viet-OCR-VQA, Viet-Receipt-VQA, Viet-Menu-VQA, Viet-Handwriting-VQA (5CD-AI) | Gemini 1.5 Flash sinh, không người duyệt; đã nằm trong dữ liệu train của Vintern-1B (paper Vintern, bảng 1) |
| VietnameseTableVQA (`YuukiAsuna`, CC BY 4.0) | Gemini 1.5 Flash sinh từ bảng Wikipedia, không người duyệt |
| AutoViVQA (arXiv 2603.09689) | Sinh tự động bằng LLM; ảnh tự nhiên |
| V-Doc (khảo sát 10/2026) | Đáp án do Qwen sinh; ảnh lấy từ Viet-Doc-VQA-II, nên rò rỉ với Vintern |

## 5. Nhóm D: không phải hỏi đáp nhưng có bbox do người gán

Dùng được cho phần định vị (ví dụ loss kéo attention của head về vùng chữ) hoặc để sinh câu hỏi tra cứu mà đáp án là trường do người gán, không phải do LLM.

| Bộ | Nội dung | Quy mô | Nhãn | Truy cập |
|---|---|---|---|---|
| **MC-OCR 2021** (RIVF 2021) | Hóa đơn tiếng Việt chụp bằng điện thoại | 2.436 ảnh (nguồn khác ghi 2.034); train có 1.155 ảnh có trường thông tin | Người gán polygon dòng chữ và tối đa 4 trường chính | Qua ban tổ chức *[cần kiểm chứng link]* |
| **UIT-DODV / UIT-DODV-Ext** (IEEE Access 2022) | Trang bài báo khoa học và sách giáo khoa, gồm bản chụp và bản scan | Bản Ext có 5.000 ảnh | Người gán bbox bảng, hình, chú thích, công thức | `uit-together.github.io/datasets`; một số bộ của UIT yêu cầu ký thỏa thuận và gửi email |
| VieBookRead (arXiv 2410.13305) | 27.025 trang sách cổ scan (1850–2003) | Lớn | Bbox từ, dòng, đoạn do Azure OCR sinh, **không phải nhãn người** | CC BY-NC 4.0, HF có cổng đăng ký |
| VinText *[cần kiểm chứng]* | Ảnh cảnh tiếng Việt, bbox mức từ | Khoảng 2.000 ảnh | Người gán | Công khai |

## 6. Tiếng Anh có bbox bằng chứng (để train chuyển ngôn ngữ)

| Bộ | Nội dung | Ghi chú |
|---|---|---|
| BBox-DocVQA (arXiv 2511.15090) | 32 nghìn câu trên 3,6 nghìn tài liệu, bbox vùng bằng chứng | Bài báo khoa học, tiếng Anh |
| BoundingDocs v2.0 | 249 nghìn câu, 8 ngôn ngữ, bbox đáp án | Một phần câu hỏi được viết lại bằng Claude; chưa rõ có tiếng Việt *[cần kiểm chứng]* |
| DocVQA (train) | Câu hỏi do người viết | Không có bbox đáp án, chỉ có OCR |

## 7. Nguồn cần xin quyền hoặc liên hệ tác giả

Thứ tự theo mức hữu ích cho đề tài:

1. **SEA-Vision** (Xiamen University, Shopee, Tongji University): xin phần tiếng Việt, gồm trang tài liệu kèm nhãn khối/dòng và câu VQA. Chưa có link công khai.
2. **ViInfographicVQA**: xin tọa độ vùng bằng chứng. Paper có bước trích vùng nhưng chỉ công bố đáp án. Nếu có tọa độ, đây sẽ là bộ train tiếng Việt có bbox lớn nhất.
3. **VisR-Bench**: xin đáp án tham chiếu và vùng bằng chứng cho phần tiếng Việt. File công khai chỉ có `page_index`. Đáp án gốc giúp đối chiếu chéo với nhãn mình đã duyệt.
4. **UIT** (nhóm UIT-Together và nhóm NLP): ký thỏa thuận để lấy UIT-DODV-Ext, ViTextVQA, OpenViVQA.
5. **Vi-ViDoRe** (GitHub `Ducduyzs`): bộ truy xuất trang tài liệu tiếng Việt nhiều lĩnh vực, có quy trình qrels do người phán xét, chưa công bố. Hỏi lịch phát hành.

Thư xin dữ liệu nên do GVHD hoặc sinh viên gửi bằng email trường, nêu rõ mục đích nghiên cứu học thuật, không phân phối lại, và trích dẫn đầy đủ.

## 8. Kết luận

- **Vẫn không có bộ hỏi đáp tài liệu tiếng Việt dày chữ nào có nhãn người ở quy mô vài nghìn câu trở lên được công bố.** Các bộ có người duyệt đều là ảnh cảnh, infographic, biểu đồ/bảng hoặc đề thi.
- **Cho ĐA2 (thiết kế 2 × 2):** ô "tiếng Việt, ảnh cảnh" dùng ViTextVQA. MTVQA tiếng Việt (884 câu test) là ứng viên phụ, sau khi kiểm tra rò rỉ.
- **Cho LoRA ở khóa luận:** gộp nhóm A và nhóm B được khoảng 80 nghìn câu tiếng Việt có người viết hoặc người duyệt (ViTextVQA 50 nghìn, ViInfographicVQA 18 nghìn, ChartTableViVQA 9 nghìn, MTVQA 4 nghìn). Nhưng **lệch miền** so với tài liệu dày chữ, và gần như không có bbox. Muốn có tài liệu dày chữ kèm bbox thì phải xin được SEA-Vision hoặc tọa độ của ViInfographicVQA.
- **Trước khi dùng bất kỳ bộ nào:** kiểm 100 mẫu ngẫu nhiên, ghi tỉ lệ lỗi; kiểm tra giấy phép; kiểm tra bộ đó có trong dữ liệu train của Vintern/Qwen2.5-VL không.

## Nguồn

- ViTextVQA: https://arxiv.org/abs/2404.10652 ; https://huggingface.co/datasets/minhquan6203/ViTextVQA
- MTVQA: https://huggingface.co/datasets/ByteDance/MTVQA
- OpenViVQA: https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset
- ViSignVQA: https://arxiv.org/html/2512.22218
- ViInfographicVQA: https://arxiv.org/html/2512.12424 ; https://huggingface.co/datasets/duytranus/ViInfographicVQA
- ChartTableViVQA: https://aclanthology.org/2026.aacl-long.5/ ; https://github.com/sonlam1102/charttablevivqa
- SEA-Vision: https://arxiv.org/abs/2603.15409
- ViExam: https://arxiv.org/abs/2508.13680
- VLSP 2025 MLQA-TSR: https://arxiv.org/html/2510.20381v1
- Viet-Doc-VQA-II: https://huggingface.co/datasets/5CD-AI/Viet-Doc-VQA-II ; Vintern-1B: https://arxiv.org/abs/2408.12480
- VietnameseTableVQA: https://huggingface.co/datasets/YuukiAsuna/VietnameseTableVQA
- AutoViVQA: https://arxiv.org/abs/2603.09689
- MC-OCR 2021: https://people.cs.umu.se/sonvx/files/MCOCR_Preprint.pdf
- UIT-DODV: https://uit-together.github.io/datasets
- VieBookRead: https://arxiv.org/abs/2410.13305 ; https://huggingface.co/datasets/thaodd11/VieBookRead
- BBox-DocVQA: https://arxiv.org/html/2511.15090v1 ; BoundingDocs v2.0 (qua): https://arxiv.org/pdf/2509.10129
- Vi-ViDoRe: https://github.com/Ducduyzs/Vi-ViDoRe-A-Governed-Multi-Domain-Benchmark-for-Vietnamese-Visual-Document-Retrieval
