import re


def clean_text(raw: str) -> str:
    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove digital signature header (first 4 lines)
        if stripped.startswith("Người ký:") or stripped.startswith("Email:"):
            continue
        if stripped.startswith("Cơ quan:") or stripped.startswith("Thời gian ký:"):
            continue
        # Remove page headers like "CÔNG BÁO/Số 363 + 364/Ngày 01-3-2024"
        if re.match(r"^CÔNG BÁO/Số\s+\d+", stripped):
            continue
        # Remove standalone page numbers
        if re.match(r"^\d{1,3}$", stripped):
            continue
        # Remove trailing reference line
        if stripped.startswith("(Xem tiếp Công báo"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)
