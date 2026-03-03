"""Query analysis for Graph RAG retrieval using Gemini structured output."""

import json
import re

from google import genai
from pydantic import BaseModel, Field

from src.config import GEMINI_API_KEY, GEMINI_MODEL


class QueryAnalysis(BaseModel):
    """Structured result of analyzing a user query before retrieval."""

    article_numbers: list[int] = Field(default_factory=list)
    chapter_references: list[str] = Field(default_factory=list)
    law_id_filter: str | None = None
    intent: str = "general"
    search_keywords: list[str] = Field(default_factory=list)
    legal_terms: list[str] = Field(default_factory=list)


def _build_query_analysis_schema() -> dict:
    """Build JSON schema for Gemini structured output."""
    return {
        "type": "object",
        "properties": {
            "article_numbers": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "chapter_references": {
                "type": "array",
                "items": {"type": "string"},
            },
            "law_id_filter": {
                "type": "string",
                "enum": ["2024", "2013"],
                "nullable": True,
            },
            "intent": {
                "type": "string",
                "enum": [
                    "comparison",
                    "definition",
                    "condition",
                    "rights_obligations",
                    "penalty",
                    "general",
                ],
            },
            "search_keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
            "legal_terms": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "article_numbers",
            "chapter_references",
            "law_id_filter",
            "intent",
            "search_keywords",
            "legal_terms",
        ],
    }


QUERY_ANALYSIS_SCHEMA = _build_query_analysis_schema()


QUERY_ANALYSIS_SYSTEM = """Bạn là chuyên gia phân tích câu hỏi pháp luật đất đai Việt Nam.

Nhiệm vụ: Phân tích câu hỏi của người dùng và trích xuất thông tin cấu trúc.

## Quy tắc phân loại intent:
- "comparison": có từ khóa so sánh, khác nhau, thay đổi, sửa đổi, bổ sung, điểm mới, so với, giữa 2013 và 2024
- "definition": hỏi về khái niệm, "là gì", "nghĩa là gì", "được hiểu như thế nào"
- "condition": hỏi về điều kiện, yêu cầu, tiêu chuẩn, "điều kiện gì", "khi nào được"
- "rights_obligations": hỏi về quyền, nghĩa vụ, trách nhiệm, "có quyền gì", "phải làm gì"
- "penalty": hỏi về xử phạt, vi phạm, chế tài, "bị phạt", "xử lý"
- "general": các câu hỏi khác

## Quy tắc trích xuất:
1. article_numbers: Trích xuất số điều từ "Điều X", "điều X", "Đ.X"
2. chapter_references: Trích xuất chương, chuyển sang số La Mã (VD: "Chương 3" -> "III", "Chương IX" giữ nguyên "IX")
3. law_id_filter: CHỈ đặt giá trị khi câu hỏi ĐẶC BIỆT đề cập đến "LĐĐ 2024", "Luật Đất đai 2024", "LĐĐ 2013", "Luật Đất đai 2013", hoặc "luật 2024/2013". Các năm khác (2025, 2026, ...) KHÔNG phải là phiên bản luật → để null. Nếu hỏi chung hoặc không rõ phiên bản → null.
4. search_keywords: 2-3 cụm từ THUẬT NGỮ PHÁP LÝ CHÍNH THỨC có trong Luật Đất đai để tìm kiếm. KHÔNG dùng từ thông dụng, KHÔNG dùng cụm từ chung chung (VD: KHÔNG viết "kinh nghiệm mua nhà", "thủ tục pháp lý", "lưu ý khi mua đất"). Chuyển đổi ngôn ngữ thông dụng sang thuật ngữ pháp lý:
   - "sổ đỏ" -> "giấy chứng nhận quyền sử dụng đất"
   - "sổ hồng" -> "giấy chứng nhận quyền sở hữu nhà ở"
   - "đền bù" -> "bồi thường"
   - "giải tỏa" -> "thu hồi đất", "giải phóng mặt bằng"
   - "sang tên" -> "chuyển nhượng quyền sử dụng đất"
   - "tách thửa" -> "tách thửa đất", "chia tách thửa đất"
   - "mua nhà đất" -> "chuyển nhượng quyền sử dụng đất", "hợp đồng chuyển nhượng"
   - "mua bán đất" -> "chuyển nhượng quyền sử dụng đất"
5. legal_terms: Các thuật ngữ pháp lý cụ thể trong câu hỏi"""


QUERY_ANALYSIS_USER = 'Phân tích câu hỏi: "{question}"'


# Mapping Arabic to Roman numerals for chapter conversion
_ARABIC_TO_ROMAN = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V",
    6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X",
    11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV",
    16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX",
}


def _regex_extract_articles(question: str) -> list[int]:
    """Pre-extract article numbers via regex as safety net."""
    matches = re.findall(r"[Đđ]iều\s+(\d+)", question)
    return [int(m) for m in matches]


def _regex_extract_chapters(question: str) -> list[str]:
    """Pre-extract chapter references and convert to Roman numerals."""
    chapters = []
    # Match "Chương" followed by Arabic number
    for m in re.findall(r"[Cc]hương\s+(\d+)", question):
        num = int(m)
        if num in _ARABIC_TO_ROMAN:
            chapters.append(_ARABIC_TO_ROMAN[num])
    # Match "Chương" followed by Roman numeral
    for m in re.findall(r"[Cc]hương\s+([IVXLCDM]+)", question):
        chapters.append(m)
    return chapters


class QueryAnalyzer:
    """Analyzes user queries using Gemini structured output before retrieval."""

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def analyze(self, question: str) -> QueryAnalysis:
        """Analyze a query using Gemini + regex fallback.

        Returns a QueryAnalysis with extracted intents, references, and keywords.
        On any error, falls back to regex-only extraction with intent="general".
        """
        # Pre-extract with regex as safety net
        regex_articles = _regex_extract_articles(question)
        regex_chapters = _regex_extract_chapters(question)

        try:
            prompt = QUERY_ANALYSIS_USER.format(question=question)
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=QUERY_ANALYSIS_SYSTEM,
                    response_mime_type="application/json",
                    response_schema=QUERY_ANALYSIS_SCHEMA,
                ),
            )

            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            data = json.loads(text.strip())

            analysis = QueryAnalysis(
                article_numbers=data.get("article_numbers", []),
                chapter_references=data.get("chapter_references", []),
                law_id_filter=data.get("law_id_filter"),
                intent=data.get("intent", "general"),
                search_keywords=data.get("search_keywords", []),
                legal_terms=data.get("legal_terms", []),
            )

            # Merge regex-extracted articles (safety net)
            merged_articles = list(
                dict.fromkeys(analysis.article_numbers + regex_articles)
            )
            analysis.article_numbers = merged_articles

            # Merge regex-extracted chapters
            merged_chapters = list(
                dict.fromkeys(analysis.chapter_references + regex_chapters)
            )
            analysis.chapter_references = merged_chapters

            return analysis

        except Exception as e:
            print(f"[QueryAnalyzer] Gemini call failed, using regex fallback: {e}")
            return QueryAnalysis(
                article_numbers=regex_articles,
                chapter_references=regex_chapters,
                intent="general",
            )
