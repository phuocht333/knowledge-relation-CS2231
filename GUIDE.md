# Graph RAG - Hỏi đáp & So sánh Luật Đất đai 2013 & 2024

Hệ thống hỏi đáp pháp luật sử dụng Knowledge Graph + Vector Search + LLM (Gemini).

## Architecture

```
PDF → Parse Articles → Extract Entities (NotebookLM/Gemini) → Build KG (NetworkX)
    → Embed (Vietnamese Sentence Transformer + FAISS) → Hybrid Retrieval
    → Q&A Agent (Gemini) → Gradio Chat UI
```

## Prerequisites

- Python 3.10+
- Google Gemini API key (https://aistudio.google.com/apikey)
- Google NotebookLM account (https://notebooklm.google.com) for entity extraction

---

## Step 1: Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
pip install pymupdf   # for PDF text extraction

# Set your Gemini API key
# Edit .env file:
echo "GEMINI_API_KEY=your-actual-key-here" > .env
```

## Step 2: Parse Articles

This parses both PDFs into structured article JSON files.

```bash
python scripts/01_parse.py
```

**Output:**
- `data/articles/articles_2024.json` — 90 articles from LĐĐ 2024
- `data/articles/articles_2013.json` — 212 articles from LĐĐ 2013
- `data/articles/articles_all.json` — Combined 302 articles

## Step 3: Extract Entities via NotebookLM

Since the Gemini free tier has rate limits, we use **Google NotebookLM** for entity extraction.

### 3a. Setup NotebookLM

1. Go to https://notebooklm.google.com
2. Create **2 notebooks** (one per law):
   - Notebook 1: Upload `docs/31-2024-qh15_1.pdf` (LĐĐ 2024)
   - Notebook 2: Upload `docs/VanBanGoc_45.2013.QH13.pdf` (LĐĐ 2013)

### 3b. Extract per chapter

For each chapter, paste the prompt below into NotebookLM (change chapter/article range accordingly).

**Prompt template:**
```
Phân tích Chương X (Điều A-B) của Luật Đất đai [2013/2024]. Trích xuất tất cả thực thể và quan hệ theo JSON format sau. CHỈ trả về JSON, không text khác.

Loại thực thể: khái_niệm, điều_luật, quyền_nghĩa_vụ, mức_hưởng, xử_phạt
Loại quan hệ: định_nghĩa, quy_định, áp_dụng, tham_chiếu, bao_gồm, điều_kiện, hạn_chế, liên_quan

{
  "entities": [
    {"id": "artX_e1", "name": "tên thực thể", "entity_type": "khái_niệm", "description": "mô tả ngắn gọn", "source_article": X, "source_text": "trích dẫn nguyên văn"}
  ],
  "relations": [
    {"source_id": "artX_e1", "target_id": "artX_e2", "relation_type": "định_nghĩa", "description": "mô tả quan hệ", "source_article": X}
  ]
}
```

### 3c. Chapter ranges & file names

**LĐĐ 2024** — Save each response to `data/entities/2024_chapter_N.json`:

| File | Chapter | Articles | Prompt change |
|------|---------|----------|---------------|
| `2024_chapter_1.json` | Chương I - Quy định chung | Điều 1-11 | Chương I (Điều 1-11) của Luật Đất đai 2024 |
| `2024_chapter_2.json` | Chương II - Quyền và trách nhiệm | Điều 12-25 | Chương II (Điều 12-25) của Luật Đất đai 2024 |
| `2024_chapter_3.json` | Chương III - Quyền và nghĩa vụ người sử dụng | Điều 26-48 | Chương III (Điều 26-48) của Luật Đất đai 2024 |
| `2024_chapter_4.json` | Chương IV - Địa giới, đo đạc, thống kê | Điều 49-59 | Chương IV (Điều 49-59) của Luật Đất đai 2024 |
| `2024_chapter_5.json` | Chương V - Quy hoạch, kế hoạch sử dụng đất | Điều 60-77 | Chương V (Điều 60-77) của Luật Đất đai 2024 |
| `2024_chapter_6.json` | Chương VI - Thu hồi đất, bồi thường | Điều 78-90 | Chương VI (Điều 78-90) của Luật Đất đai 2024 |

**LĐĐ 2013** — Save each response to `data/entities/2013_chapter_N.json`:

| File | Chapter | Articles | Prompt change |
|------|---------|----------|---------------|
| `2013_chapter_1.json` | Chương I - Quy định chung | Điều 1-12 | Chương I (Điều 1-12) của Luật Đất đai 2013 |
| `2013_chapter_2.json` | Chương II - Quyền và trách nhiệm NN | Điều 13-28 | Chương II (Điều 13-28) của Luật Đất đai 2013 |
| `2013_chapter_3.json` | Chương III - Địa giới hành chính | Điều 29-34 | Chương III (Điều 29-34) của Luật Đất đai 2013 |
| `2013_chapter_4.json` | Chương IV - Quy hoạch, kế hoạch SDĐ | Điều 35-51 | Chương IV (Điều 35-51) của Luật Đất đai 2013 |
| `2013_chapter_5.json` | Chương V - Giao đất, cho thuê đất | Điều 52-60 | Chương V (Điều 52-60) của Luật Đất đai 2013 |
| `2013_chapter_6.json` | Chương VI - Thu hồi đất, bồi thường | Điều 61-94 | Chương VI (Điều 61-94) của Luật Đất đai 2013 |
| `2013_chapter_7.json` | Chương VII - Đăng ký đất đai | Điều 95-106 | Chương VII (Điều 95-106) của Luật Đất đai 2013 |
| `2013_chapter_8.json` | Chương VIII - Tài chính, giá đất | Điều 107-119 | Chương VIII (Điều 107-119) của Luật Đất đai 2013 |
| `2013_chapter_9.json` | Chương IX - Hệ thống thông tin đất đai | Điều 120-124 | Chương IX (Điều 120-124) của Luật Đất đai 2013 |
| `2013_chapter_10.json` | Chương X - Chế độ sử dụng các loại đất | Điều 125-165 | Chương X (Điều 125-165) của Luật Đất đai 2013 |
| `2013_chapter_11.json` | Chương XI - Quyền và nghĩa vụ NSD đất | Điều 166-194 | Chương XI (Điều 166-194) của Luật Đất đai 2013 |
| `2013_chapter_12.json` | Chương XII - Thủ tục hành chính | Điều 195-197 | Chương XII (Điều 195-197) của Luật Đất đai 2013 |
| `2013_chapter_13.json` | Chương XIII - Giám sát, thanh tra | Điều 198-209 | Chương XIII (Điều 198-209) của Luật Đất đai 2013 |
| `2013_chapter_14.json` | Chương XIV - Điều khoản thi hành | Điều 210-212 | Chương XIV (Điều 210-212) của Luật Đất đai 2013 |

### 3d. Merge all extractions

```bash
python scripts/merge_notebooklm.py
```

**Output:**
- `data/entities/entities_2024.json`
- `data/entities/relations_2024.json`
- `data/entities/entities_2013.json`
- `data/entities/relations_2013.json`

### Alternative: Extract via Gemini API

If your Gemini API has quota, you can skip NotebookLM and run:

```bash
python scripts/02_extract.py
```

This calls Gemini API for all 302 articles automatically (takes ~10 min).

## Step 4: Build Knowledge Graph

```bash
python scripts/03_build_graph.py
```

This creates a NetworkX graph with:
- Article nodes (302) + Entity nodes + Chapter nodes
- Intra-law edges (references, definitions)
- **Cross-version edges** linking related articles between 2013 & 2024

**Output:** `data/graph/knowledge_graph.json`

## Step 5: Compute Embeddings

```bash
python scripts/04_embed.py
```

Downloads the `dangvantuan/vietnamese-embedding` model (~500MB first time) and embeds all articles, entities, and relations.

**Output:** `data/embeddings/index.faiss` + `data/embeddings/metadata.json`

## Step 6: Launch Q&A Interface

```bash
python -m src.app
```

Open http://localhost:7860 in your browser.

### Example Questions

**Single law:**
- "Nhà nước thu hồi đất trong trường hợp nào?" → Cites Điều 78, 79, 81 (LĐĐ 2024)
- "Bản đồ địa chính là gì?" → Cites Điều 3 khoản 1
- "Quyền của người sử dụng đất?" → Cites Điều 26, 27

**Comparison between 2013 & 2024:**
- "So sánh quy định thu hồi đất giữa LĐĐ 2013 và 2024"
- "Điểm mới của Luật Đất đai 2024 so với 2013?"
- "Phân loại đất thay đổi như thế nào giữa 2 phiên bản?"

---

## Quick Reference: Full Pipeline

```bash
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install pymupdf
echo "GEMINI_API_KEY=your-key" > .env

# Pipeline
python scripts/01_parse.py              # Parse PDFs → articles
# ... do NotebookLM extraction ...      # Extract entities (manual)
python scripts/merge_notebooklm.py      # Merge chapter JSONs
python scripts/03_build_graph.py        # Build knowledge graph
python scripts/04_embed.py              # Compute embeddings
python -m src.app                       # Launch Gradio UI
```

## Project Structure

```
graphRAG/
├── .env                          # GEMINI_API_KEY
├── requirements.txt
├── GUIDE.md                      # This file
├── docs/
│   ├── 31-2024-qh15_1.pdf       # LĐĐ 2024 source
│   ├── 31-2024-qh15_1.txt       # Extracted text
│   ├── VanBanGoc_45.2013.QH13.pdf  # LĐĐ 2013 source
│   └── VanBanGoc_45.2013.QH13.txt  # Extracted text
├── src/
│   ├── config.py                 # Settings, paths, law definitions
│   ├── parsing/                  # PDF text → structured articles
│   ├── extraction/               # Entity/relation extraction (Gemini)
│   ├── graph/                    # NetworkX knowledge graph
│   ├── embedding/                # Vietnamese embeddings + FAISS
│   ├── retrieval/                # Hybrid graph + vector retrieval
│   ├── qa/                       # Gemini Q&A agent with citations
│   └── app.py                   # Gradio chat UI
├── scripts/
│   ├── 01_parse.py               # Parse both law PDFs
│   ├── 02_extract.py             # Extract entities via Gemini API
│   ├── 03_build_graph.py         # Build knowledge graph
│   ├── 04_embed.py               # Compute embeddings
│   ├── merge_notebooklm.py       # Merge NotebookLM outputs
│   └── run_pipeline.py           # Run steps 1-4 automatically
└── data/
    ├── articles/                 # Parsed articles JSON
    ├── entities/                 # Extracted entities + NotebookLM chapters
    ├── graph/                    # Serialized knowledge graph
    └── embeddings/               # FAISS index + metadata
```
