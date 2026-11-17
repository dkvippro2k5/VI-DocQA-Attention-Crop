"""Prompt sinh đáp án dùng chung cho MỌI nhánh (A, B, C1-C5 và các nhánh đối chứng).

Chốt ngày 2026-11-11, trước khi chạy nhánh nào trên bộ nhãn mới (xem scope mục 9). Đổi prompt thì phải chạy lại mọi
nhánh đã có kết quả (CLAUDE.md).

- Một prompt cho mọi câu: không chọn prompt theo question_type hay answer_form, vì đó là thông tin nhãn.
- Khớp quy tắc gán nhãn: chép nguyên văn, ngắn nhất mà đủ ý, câu nhiều ý ngăn bằng dấu chấm phẩy (để chấm item_cover).
- MAX_NEW_TOKENS = 384: đáp án nhiều ý ở vòng 2 có trung vị 41 từ, 10% dài hơn 139 từ; trần cũ 64 token cắt cụt.
Nhánh A (LLM đọc text) dùng cùng ANSWER_INSTRUCTION, chỉ thay ảnh bằng text ngữ cảnh.
"""

PROMPT_VERSION = "p2"  # p1: "Trả lời ngắn gọn bằng một cụm từ hoặc một câu.", max_new_tokens 64 (C3 dev tháng 10)

ANSWER_INSTRUCTION = (
    "Trả lời câu hỏi dựa vào tài liệu. Chép nguyên văn từ tài liệu, ngắn nhất mà vẫn đủ ý. "
    "Nếu câu hỏi yêu cầu nhiều ý (liệt kê, các bước, các điều kiện, các thành phần), "
    "nêu đủ mọi ý, mỗi ý ngắn gọn, ngăn cách bằng dấu chấm phẩy. "
    "Chỉ ghi đáp án, không giải thích."
)
ANSWER_SUFFIX = "\n" + ANSWER_INSTRUCTION  # nối sau câu hỏi
MAX_NEW_TOKENS = 384

# Bản tiếng Anh cho TextVQA, DocVQA trong thiết kế 2 × 2 (chốt 15/11/2026): dịch sát từng ý của ANSWER_INSTRUCTION,
# giữ "document" như bản tiếng Việt dùng "tài liệu" cho cả ảnh cảnh (ViTextVQA), để hai ngôn ngữ chỉ khác nhau ở ngôn ngữ.
ANSWER_INSTRUCTION_EN = (
    "Answer the question based on the document. Copy the wording from the document verbatim, as short as possible "
    "while still complete. If the question asks for multiple items (a list, steps, conditions, components), "
    "give every item, each brief, separated by semicolons. "
    "Write only the answer, no explanation."
)
ANSWER_SUFFIX_EN = "\n" + ANSWER_INSTRUCTION_EN
EN_DATASETS = {"textvqa", "docvqa"}


def answer_suffix(dataset: str) -> str:
    """Câu chỉ dẫn theo ngôn ngữ của bộ dữ liệu; bộ tiếng Việt (vitextvqa, visr_bench_vi) dùng bản gốc."""
    return ANSWER_SUFFIX_EN if dataset in EN_DATASETS else ANSWER_SUFFIX
