import re
import networkx as nx

from src.extraction.models import Entity, Relation
from src.parsing.models import Article


# Mapping of related topics between 2013 and 2024 for cross-version links
CROSS_VERSION_TOPIC_MAP = {
    # (2013 article range, 2024 article range, topic)
    "phạm vi điều chỉnh": ([1], [1]),
    "đối tượng áp dụng": ([2], [2]),
    "giải thích từ ngữ": ([3], [3]),
    "người sử dụng đất": ([5], [4]),
    "phân loại đất": ([10], [9]),
    "hành vi bị nghiêm cấm": ([12], [11]),
    "sở hữu đất đai": ([4], [12]),
    "quyền của nhà nước": ([13, 14], [13, 14]),
    "quyền chung người sử dụng": ([166], [26]),
    "nghĩa vụ chung": ([170], [31]),
    "thu hồi đất quốc phòng": ([61, 62], [78, 79]),
    "thu hồi đất vi phạm": ([64], [81]),
    "quy hoạch sử dụng đất": ([35, 36, 37], [60, 61, 62]),
    "bồi thường thu hồi": ([74, 75], [86, 87]),
    "điều kiện chuyển nhượng": ([188], [45]),
    "cưỡng chế thu hồi": ([71], [89]),
}


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build(
        self,
        entities: list[Entity],
        relations: list[Relation],
        articles: list[Article],
    ) -> nx.DiGraph:
        self.graph.clear()

        # Add entity nodes
        for e in entities:
            self.graph.add_node(e.id, **{
                "name": e.name,
                "entity_type": e.entity_type.value,
                "description": e.description,
                "source_article": e.source_article,
                "source_text": e.source_text,
                "law_id": e.law_id,
                "node_type": "entity",
            })

        # Add article nodes
        articles_by_key: dict[tuple[str, int], Article] = {}
        for art in articles:
            node_id = f"{art.law_id}_article_{art.article_number}"
            articles_by_key[(art.law_id, art.article_number)] = art
            self.graph.add_node(node_id, **{
                "name": f"Điều {art.article_number}",
                "title": art.title,
                "chapter": art.chapter,
                "chapter_title": art.chapter_title,
                "section": art.section or "",
                "section_title": art.section_title or "",
                "content": art.content[:500],
                "law_id": art.law_id,
                "node_type": "article",
            })

        # Add extracted relations
        for r in relations:
            if self.graph.has_node(r.source_id) and self.graph.has_node(r.target_id):
                self.graph.add_edge(r.source_id, r.target_id, **{
                    "relation_type": r.relation_type.value,
                    "description": r.description,
                    "source_article": r.source_article,
                    "law_id": r.law_id,
                })

        # Structural edges: entity -> article
        for e in entities:
            art_node = f"{e.law_id}_article_{e.source_article}"
            if self.graph.has_node(art_node):
                self.graph.add_edge(e.id, art_node, **{
                    "relation_type": "thuộc_điều",
                    "description": f"{e.name} thuộc Điều {e.source_article} (LĐĐ {e.law_id})",
                    "source_article": e.source_article,
                    "law_id": e.law_id,
                })

        # Structural edges: article -> chapter
        chapters_seen = set()
        for art in articles:
            ch_node = f"{art.law_id}_chapter_{art.chapter}"
            if ch_node not in chapters_seen:
                self.graph.add_node(ch_node, **{
                    "name": art.chapter,
                    "title": art.chapter_title,
                    "law_id": art.law_id,
                    "node_type": "chapter",
                })
                chapters_seen.add(ch_node)
            art_node = f"{art.law_id}_article_{art.article_number}"
            self.graph.add_edge(art_node, ch_node, **{
                "relation_type": "thuộc_chương",
                "description": f"Điều {art.article_number} thuộc {art.chapter} (LĐĐ {art.law_id})",
                "source_article": art.article_number,
                "law_id": art.law_id,
            })

        # Cross-reference edges within same law
        for art in articles:
            refs = re.findall(r"(?:theo|tại|quy định tại)\s+Điều\s+(\d+)", art.content)
            art_node = f"{art.law_id}_article_{art.article_number}"
            for ref_num in set(refs):
                ref_num = int(ref_num)
                if ref_num != art.article_number:
                    ref_node = f"{art.law_id}_article_{ref_num}"
                    if self.graph.has_node(ref_node):
                        self.graph.add_edge(art_node, ref_node, **{
                            "relation_type": "tham_chiếu",
                            "description": f"Điều {art.article_number} tham chiếu Điều {ref_num} (LĐĐ {art.law_id})",
                            "source_article": art.article_number,
                            "law_id": art.law_id,
                        })

        # Cross-version edges: link related articles between 2013 and 2024
        self._add_cross_version_edges(articles_by_key)

        # Definition usage edges for both laws
        for law_id in ["2013", "2024"]:
            self._add_definition_usage_edges(entities, articles, law_id, articles_by_key)

        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    def _add_cross_version_edges(self, articles_by_key: dict) -> None:
        for topic, (arts_2013, arts_2024) in CROSS_VERSION_TOPIC_MAP.items():
            for a13 in arts_2013:
                for a24 in arts_2024:
                    node_13 = f"2013_article_{a13}"
                    node_24 = f"2024_article_{a24}"
                    if self.graph.has_node(node_13) and self.graph.has_node(node_24):
                        self.graph.add_edge(node_13, node_24, **{
                            "relation_type": "phiên_bản_mới",
                            "description": f"Điều {a13} (LĐĐ 2013) -> Điều {a24} (LĐĐ 2024): {topic}",
                            "source_article": a13,
                            "law_id": "cross",
                        })
                        self.graph.add_edge(node_24, node_13, **{
                            "relation_type": "phiên_bản_cũ",
                            "description": f"Điều {a24} (LĐĐ 2024) <- Điều {a13} (LĐĐ 2013): {topic}",
                            "source_article": a24,
                            "law_id": "cross",
                        })

        # Also link articles with same number (common pattern in VN law revisions)
        all_2013 = {k[1] for k in articles_by_key if k[0] == "2013"}
        all_2024 = {k[1] for k in articles_by_key if k[0] == "2024"}
        common = all_2013 & all_2024
        for art_num in common:
            node_13 = f"2013_article_{art_num}"
            node_24 = f"2024_article_{art_num}"
            if not self.graph.has_edge(node_13, node_24):
                self.graph.add_edge(node_13, node_24, **{
                    "relation_type": "cùng_số_điều",
                    "description": f"Điều {art_num}: so sánh LĐĐ 2013 vs LĐĐ 2024",
                    "source_article": art_num,
                    "law_id": "cross",
                })

    def _add_definition_usage_edges(
        self,
        entities: list[Entity],
        articles: list[Article],
        law_id: str,
        articles_by_key: dict,
    ) -> None:
        art3 = articles_by_key.get((law_id, 3))
        if not art3:
            return

        terms = re.findall(r"\d+\.\s+(.+?)\s+là\s+", art3.content)
        law_entities = [e for e in entities if e.law_id == law_id and e.source_article == 3]

        for term in terms:
            term_lower = term.lower().strip()
            for other_art in articles:
                if other_art.law_id != law_id or other_art.article_number == 3:
                    continue
                if term_lower in other_art.content.lower():
                    for e in law_entities:
                        if term_lower in e.name.lower():
                            other_node = f"{law_id}_article_{other_art.article_number}"
                            if not self.graph.has_edge(e.id, other_node):
                                self.graph.add_edge(e.id, other_node, **{
                                    "relation_type": "được_sử_dụng",
                                    "description": f"Khái niệm '{e.name}' được sử dụng trong Điều {other_art.article_number} (LĐĐ {law_id})",
                                    "source_article": 3,
                                    "law_id": law_id,
                                })
                            break
