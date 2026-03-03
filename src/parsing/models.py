from pydantic import BaseModel


class Article(BaseModel):
    article_number: int
    title: str
    chapter: str
    chapter_title: str
    section: str | None = None
    section_title: str | None = None
    content: str
    start_line: int
    end_line: int
    law_id: str = "2024"  # "2024" or "2013"

    @property
    def full_id(self) -> str:
        return f"{self.law_id}_Điều_{self.article_number}"

    @property
    def display_id(self) -> str:
        return f"Điều {self.article_number} (LĐĐ {self.law_id})"

    @property
    def summary_header(self) -> str:
        parts = [f"Điều {self.article_number}. {self.title}"]
        parts.append(f"({self.chapter} - {self.chapter_title})")
        if self.section:
            parts.append(f"[{self.section} - {self.section_title}]")
        parts.append(f"[Luật Đất đai {self.law_id}]")
        return " ".join(parts)
