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

### 5 loại thực thể (Entity Types)

| Loại | Ý nghĩa | Ví dụ |
|---|---|---|
| `khái_niệm` | Thuật ngữ, định nghĩa pháp lý | "chế độ sở hữu đất đai", "thống nhất quản lý về đất đai" |
| `điều_luật` | Điều khoản cụ thể trong luật | "Điều 1 - Phạm vi điều chỉnh" |
| `quyền_nghĩa_vụ` | Quyền lợi & trách nhiệm | "quyền và nghĩa vụ của công dân, người sử dụng đất" |
| `mức_hưởng` | Ưu đãi, bồi thường | "miễn, giảm tiền sử dụng đất", "bồi thường, hỗ trợ, tái định cư" |
| `xử_phạt` | Chế tài, xử lý vi phạm | "xử lý vi phạm pháp luật về đất đai", "Nhà nước thu hồi đất" |

### 8 loại quan hệ nội luật (Relation Types)

| Loại | Ý nghĩa | Ví dụ |
|---|---|---|
| `quy_định` | A quy định nội dung B | LĐĐ 2024 —quy_định→ chế độ sở hữu đất đai |
| `định_nghĩa` | A định nghĩa khái niệm B | Điều 3 —định_nghĩa→ "bản đồ địa chính" |
| `áp_dụng` | A áp dụng cho đối tượng B | LĐĐ 2024 —áp_dụng→ đất đai thuộc lãnh thổ VN |
| `điều_kiện` | A là điều kiện để B | sử dụng đất ổn định —điều_kiện→ được công nhận người sử dụng đất |
| `tham_chiếu` | A tham chiếu đến B | Điều 36 —tham_chiếu→ Điều 78 (thu hồi đất) |
| `bao_gồm` | A bao gồm B | Chương I —bao_gồm→ Điều 1, 2, ... |
| `hạn_chế` | A hạn chế quyền B | — |
| `liên_quan` | A liên quan đến B | quyền hạn NN —liên_quan→ đại diện chủ sở hữu toàn dân |

### Liên kết chéo giữa LĐĐ 2013 ↔ 2024 (151 cross-version edges)

Đây là điểm **khác biệt cốt lõi** so với RAG truyền thống — graph kết nối trực tiếp các điều luật tương ứng giữa 2 phiên bản.

| Loại quan hệ | Số lượng | Ý nghĩa |
|---|---|---|
| `cùng_số_điều` | 85 | Điều cùng số giữa 2 phiên bản |
| `phiên_bản_cũ` | 33 | Điều 2024 → trỏ về điều tương ứng trong 2013 |
| `phiên_bản_mới` | 33 | Điều 2013 → trỏ đến điều tương ứng trong 2024 |

**Ví dụ:**
```
Điều 1 (LĐĐ 2024) ──phiên_bản_cũ──→ Điều 1 (LĐĐ 2013)    [phạm vi điều chỉnh]
Điều 4 (LĐĐ 2024) ──phiên_bản_cũ──→ Điều 5 (LĐĐ 2013)    [người sử dụng đất — khác số điều]
Điều 9 (LĐĐ 2024) ──phiên_bản_cũ──→ Điều 10 (LĐĐ 2013)   [phân loại đất — khác số điều]
```

> Khi hỏi so sánh, hệ thống traverse từ điều 2024 → sang điều tương ứng 2013 (và ngược lại) để lấy ngữ cảnh cả 2 bên.

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
