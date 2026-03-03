import re
from src.parsing.text_cleaner import clean_text
from src.parsing.models import Article


def parse_articles(raw_text: str, law_id: str = "2024") -> list[Article]:
    text = clean_text(raw_text)
    lines = text.split("\n")

    # Pass 1: Build structural index
    chapters: list[dict] = []
    sections: list[dict] = []
    article_markers: list[dict] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Chapter: "Chương I", "Chương II", etc.
        ch_match = re.match(r"^Chương\s+([IVX]+)\s*$", stripped)
        if ch_match:
            # Next non-empty line is the chapter title
            title = ""
            for j in range(i + 1, min(i + 3, len(lines))):
                t = lines[j].strip()
                if t:
                    title = t
                    break
            chapters.append({
                "number": ch_match.group(1),
                "title": title,
                "line": i,
            })
            continue

        # Section: "Mục 1", "Mục 2", etc.
        sec_match = re.match(r"^Mục\s+(\d+)\s*$", stripped)
        if sec_match:
            title = ""
            for j in range(i + 1, min(i + 3, len(lines))):
                t = lines[j].strip()
                if t:
                    title = t
                    break
            sections.append({
                "number": sec_match.group(1),
                "title": title,
                "line": i,
            })
            continue

        # Article: "Điều 1. Phạm vi điều chỉnh"
        art_match = re.match(r"^Điều\s+(\d+)\.\s+(.+)$", stripped)
        if art_match:
            article_markers.append({
                "number": int(art_match.group(1)),
                "title": art_match.group(2).strip(),
                "line": i,
            })

    # Pass 2: Extract content between article boundaries
    articles = []
    for idx, marker in enumerate(article_markers):
        start_line = marker["line"]
        if idx + 1 < len(article_markers):
            end_line = article_markers[idx + 1]["line"]
        else:
            end_line = len(lines)

        # Collect content lines (skip the article header line itself)
        content_lines = []
        for k in range(start_line + 1, end_line):
            stripped = lines[k].strip()
            # Stop at next chapter/section header
            if re.match(r"^Chương\s+[IVX]+\s*$", stripped):
                break
            if re.match(r"^Mục\s+\d+\s*$", stripped):
                break
            content_lines.append(lines[k])

        content = "\n".join(content_lines).strip()
        # Clean up excessive blank lines
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Determine chapter and section for this article
        chapter_num = ""
        chapter_title = ""
        for ch in chapters:
            if ch["line"] <= start_line:
                chapter_num = f"Chương {ch['number']}"
                chapter_title = ch["title"]

        section_num = None
        section_title = None
        for sec in sections:
            if sec["line"] <= start_line:
                # Only assign if section is within current chapter
                sec_chapter = ""
                for ch in chapters:
                    if ch["line"] <= sec["line"]:
                        sec_chapter = ch["number"]
                current_chapter = ""
                for ch in chapters:
                    if ch["line"] <= start_line:
                        current_chapter = ch["number"]
                if sec_chapter == current_chapter:
                    section_num = f"Mục {sec['number']}"
                    section_title = sec["title"]

        articles.append(Article(
            article_number=marker["number"],
            title=marker["title"],
            chapter=chapter_num,
            chapter_title=chapter_title,
            section=section_num,
            section_title=section_title,
            content=content,
            start_line=start_line,
            end_line=end_line,
            law_id=law_id,
        ))

    return articles
