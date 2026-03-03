import json
from collections import defaultdict

import networkx as nx

from src.config import TOP_K_VECTOR, GRAPH_HOP_DEPTH, TOP_K_FINAL, ARTICLES_DIR
from src.embedding.text_embedder import TextEmbedder
from src.embedding.vector_store import VectorStore
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.subgraph_extractor import format_subgraph_context

# Intent-to-entity-type boosting map
_INTENT_ENTITY_BOOST = {
    "definition": "khái_niệm",
    "rights_obligations": "quyền_nghĩa_vụ",
    "penalty": "xử_phạt",
}


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        graph: nx.DiGraph,
        embedder: TextEmbedder,
    ):
        self.vector_store = vector_store
        self.graph = graph
        self.embedder = embedder
        self.query_analyzer = QueryAnalyzer()
        self._articles_content = self._load_articles_content()

    def _load_articles_content(self) -> dict[str, str]:
        """Load all articles content. Keys: '{law_id}_{article_number}'"""
        content = {}
        for filename in ["articles_2024.json", "articles_2013.json", "articles.json"]:
            path = ARTICLES_DIR / filename
            if not path.exists():
                continue
            articles = json.loads(path.read_text(encoding="utf-8"))
            for a in articles:
                law_id = a.get("law_id", "2024")
                key = f"{law_id}_{a['article_number']}"
                content[key] = a["content"]
        return content

    def _is_comparison_query(self, query: str) -> bool:
        comparison_keywords = [
            "so sánh", "khác nhau", "thay đổi", "sửa đổi", "bổ sung",
            "khác biệt", "mới", "cũ", "2013", "2024", "hai luật",
            "hai phiên bản", "điểm mới", "so với",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in comparison_keywords)

    def _inject_article_nodes(
        self,
        analysis,
        candidate_nodes: set,
        node_scores: dict[str, float],
    ) -> None:
        """Directly inject article/chapter nodes from query analysis."""
        law_ids = [analysis.law_id_filter] if analysis.law_id_filter else ["2024", "2013"]

        # Inject article nodes
        for art_num in analysis.article_numbers:
            for law_id in law_ids:
                art_node = f"{law_id}_article_{art_num}"
                if self.graph.has_node(art_node):
                    candidate_nodes.add(art_node)
                    node_scores[art_node] = max(node_scores.get(art_node, 0), 1.0)

        # Inject chapter nodes: find all articles belonging to the chapter
        for chapter_ref in analysis.chapter_references:
            for law_id in law_ids:
                for node_id, attrs in self.graph.nodes(data=True):
                    if (
                        attrs.get("node_type") == "article"
                        and attrs.get("law_id", "2024") == law_id
                        and attrs.get("chapter", "").startswith(f"Chương {chapter_ref}")
                    ):
                        candidate_nodes.add(node_id)
                        node_scores[node_id] = max(
                            node_scores.get(node_id, 0), 0.95
                        )

    def _multi_query_vector_search(
        self,
        query: str,
        analysis,
        top_k: int,
    ) -> list[dict]:
        """Run vector search on original query + expanded keywords."""
        # Original query search
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Additional searches for expanded keywords
        keyword_top_k = max(1, TOP_K_VECTOR // 2)
        seen_ids = {r.get("id", "") for r in results}
        # Use the minimum original result score as a quality threshold
        orig_scores = [r.get("score", 0.0) for r in results if r.get("score", 0.0) > 0]
        min_score_threshold = min(orig_scores) * 0.85 if orig_scores else 0.0

        for keyword in analysis.search_keywords:
            kw_embedding = self.embedder.embed_query(keyword)
            kw_results = self.vector_store.search(kw_embedding, top_k=keyword_top_k)
            for r in kw_results:
                rid = r.get("id", "")
                raw_score = r.get("score", 0.0)
                if rid not in seen_ids and raw_score >= min_score_threshold:
                    r["score"] = raw_score * 0.8  # discount
                    results.append(r)
                    seen_ids.add(rid)

        return results

    def _apply_analysis_boosting(
        self,
        analysis,
        node_scores: dict[str, float],
        all_nodes: set,
    ) -> None:
        """Apply law version filtering and intent-based entity type boosting."""
        # Law version boosting
        if analysis.law_id_filter:
            for node in all_nodes:
                if not self.graph.has_node(node):
                    continue
                node_law = self.graph.nodes[node].get("law_id", "2024")
                if node_law == analysis.law_id_filter:
                    node_scores[node] *= 1.2
                elif node_law in ("2024", "2013"):
                    node_scores[node] *= 0.7

        # Intent-to-entity-type boosting
        boost_type = _INTENT_ENTITY_BOOST.get(analysis.intent)
        if boost_type:
            for node in all_nodes:
                if not self.graph.has_node(node):
                    continue
                attrs = self.graph.nodes[node]
                if attrs.get("entity_type") == boost_type:
                    node_scores[node] *= 1.3

    def retrieve(self, query: str) -> dict:
        # Step 0: Query Analysis
        analysis = self.query_analyzer.analyze(query)
        is_comparison = analysis.intent == "comparison" or self._is_comparison_query(query)

        # Step 0.5: Direct Node Injection
        candidate_nodes = set()
        node_scores: dict[str, float] = defaultdict(float)
        self._inject_article_nodes(analysis, candidate_nodes, node_scores)

        # Step 1: Multi-Query Vector Search
        top_k = TOP_K_VECTOR * 2 if is_comparison else TOP_K_VECTOR
        vector_results = self._multi_query_vector_search(query, analysis, top_k)

        # Step 2: Collect candidate node IDs from vector results
        for result in vector_results:
            node_id = result.get("id", "")
            score = result.get("score", 0.0)

            if node_id and self.graph.has_node(node_id):
                candidate_nodes.add(node_id)
                node_scores[node_id] = max(node_scores[node_id], score)

            # Also add article node for entities/relations
            art_num = result.get("article_number") or result.get("source_article")
            law_id = result.get("law_id", "2024")
            if art_num:
                art_node = f"{law_id}_article_{art_num}"
                if self.graph.has_node(art_node):
                    candidate_nodes.add(art_node)
                    node_scores[art_node] = max(node_scores[art_node], score * 0.9)

        # Step 3: For comparison queries, also pull cross-version nodes
        if is_comparison:
            cross_nodes = set()
            for node in list(candidate_nodes):
                if not self.graph.has_node(node):
                    continue
                for _, tgt, attrs in self.graph.out_edges(node, data=True):
                    if attrs.get("law_id") == "cross":
                        cross_nodes.add(tgt)
                        node_scores[tgt] = max(node_scores.get(tgt, 0), node_scores[node] * 0.85)
                for src, _, attrs in self.graph.in_edges(node, data=True):
                    if attrs.get("law_id") == "cross":
                        cross_nodes.add(src)
                        node_scores[src] = max(node_scores.get(src, 0), node_scores[node] * 0.85)
            candidate_nodes |= cross_nodes

        # Step 4: Graph expansion (BFS 1-2 hops)
        expanded_nodes = set()
        for node in list(candidate_nodes):
            try:
                neighbors = nx.single_source_shortest_path_length(
                    self.graph, node, cutoff=GRAPH_HOP_DEPTH
                )
                for neighbor, dist in neighbors.items():
                    if neighbor not in candidate_nodes:
                        expanded_nodes.add(neighbor)
                        base_score = node_scores.get(node, 0.5)
                        node_scores[neighbor] = max(
                            node_scores.get(neighbor, 0),
                            base_score * (0.5 ** dist),
                        )
            except nx.NetworkXError:
                pass

        all_nodes = candidate_nodes | expanded_nodes

        # Step 5: Re-rank
        for node in all_nodes:
            if self.graph.has_node(node):
                neighbors_in_candidates = sum(
                    1 for n in self.graph.neighbors(node)
                    if n in candidate_nodes
                )
                node_scores[node] += neighbors_in_candidates * 0.1

        # Step 5.5: Apply analysis-based boosting (law version + entity type)
        self._apply_analysis_boosting(analysis, node_scores, all_nodes)

        # Step 6: Select top-K nodes
        final_k = TOP_K_FINAL * 4 if is_comparison else TOP_K_FINAL * 3
        ranked = sorted(all_nodes, key=lambda n: node_scores.get(n, 0), reverse=True)
        top_nodes = ranked[:final_k]

        # Format context
        context = format_subgraph_context(
            self.graph, top_nodes, self._articles_content
        )

        # Collect cited articles
        cited_articles = {}
        for nid in top_nodes:
            if "_article_" in nid:
                parts = nid.split("_article_")
                if len(parts) == 2:
                    law_id = parts[0]
                    art_num = int(parts[1])
                    label = f"Điều {art_num} (LĐĐ {law_id})"
                    cited_articles[label] = (law_id, art_num)
            elif self.graph.has_node(nid):
                attrs = self.graph.nodes[nid]
                art = attrs.get("source_article")
                law_id = attrs.get("law_id", "2024")
                if art:
                    label = f"Điều {art} (LĐĐ {law_id})"
                    cited_articles[label] = (law_id, art)

        return {
            "context": context,
            "cited_articles": sorted(cited_articles.keys()),
            "is_comparison": is_comparison,
            "vector_results": vector_results[:5],
            "num_nodes_retrieved": len(top_nodes),
            "query_analysis": analysis.model_dump(),
        }
