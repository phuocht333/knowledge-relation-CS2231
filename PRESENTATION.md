
# Graph RAG — Hỏi đáp Luật Đất đai 2013 & 2024

> Knowledge Graph + Vector Search + LLM → Tra cứu & so sánh luật thông minh

---

## Vấn đề

- Luật Đất đai 2024 thay thế 2013 → nhiều thay đổi, khó tra cứu thủ công
- RAG truyền thống (chỉ vector search) → thiếu **quan hệ ngữ nghĩa** giữa các điều luật
- Không hỗ trợ **so sánh xuyên phiên bản**

---

## Giải pháp: Graph RAG

```
PDF → Parse 302 điều luật → Trích xuất thực thể (NotebookLM/Gemini)
    → Knowledge Graph (4,379 nodes, 11,793 edges)
    → Embedding tiếng Việt (FAISS, 11,448 vectors)
    → Hybrid Retrieval (Vector + Graph) → Q&A Agent (Gemini)
```

---

## Hybrid Retrieval — Điểm khác biệt chính

| | RAG truyền thống | Graph RAG (dự án này) |
|---|---|---|
| Tìm kiếm | Vector search | Vector search **+ Graph traversal** |
| Quan hệ giữa điều luật | Không | Có (8 loại quan hệ) |
| So sánh 2013 ↔ 2024 | Không | Có (cross-version edges) |
| Phân tích query | Không | Query Analyzer (Gemini) |

---

## Knowledge Graph

- **4,379 nodes** — điều luật, chương, thực thể pháp lý
- **11,793 edges** — quan hệ nội luật + liên kết chéo 2013 ↔ 2024
- 5 loại thực thể: khái niệm, điều luật, quyền/nghĩa vụ, mức hưởng, xử phạt
- 8 loại quan hệ: định nghĩa, quy định, áp dụng, tham chiếu, bao gồm, điều kiện, hạn chế, liên quan

---

## Tech Stack

| Công nghệ | Vai trò |
|---|---|
| Gemini | LLM (Q&A, trích xuất, phân tích query) |
| NetworkX | Đồ thị tri thức |
| FAISS | Vector search |
| Vietnamese Sentence Transformer | Embedding tiếng Việt |
| Gradio | Giao diện web |

---

## Demo

- Chat hỏi đáp (hỗ trợ hội thoại đa lượt)
- So sánh quy định giữa LĐĐ 2013 và 2024
- Khám phá đồ thị tri thức (interactive visualization)

```
http://localhost:7860
```
