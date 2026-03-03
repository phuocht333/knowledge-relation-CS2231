import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.config import ARTICLES_DIR, ENTITIES_DIR, GRAPH_DIR, LAWS
from src.parsing.models import Article
from src.extraction.models import Entity, Relation
from src.graph.kg_builder import KnowledgeGraphBuilder
from src.graph.kg_store import save_graph


def main():
    print("=== Step 3: Building knowledge graph (both laws) ===")

    all_articles = []
    all_entities = []
    all_relations = []

    for law_id in LAWS:
        # Load articles
        art_path = ARTICLES_DIR / f"articles_{law_id}.json"
        if art_path.exists():
            arts = [Article(**a) for a in json.loads(art_path.read_text(encoding="utf-8"))]
            all_articles.extend(arts)
            print(f"  {LAWS[law_id]['name']}: {len(arts)} articles")

        # Load entities
        ent_path = ENTITIES_DIR / f"entities_{law_id}.json"
        if ent_path.exists():
            ents = [Entity(**e) for e in json.loads(ent_path.read_text(encoding="utf-8"))]
            all_entities.extend(ents)
            print(f"    {len(ents)} entities")

        # Load relations
        rel_path = ENTITIES_DIR / f"relations_{law_id}.json"
        if rel_path.exists():
            rels = [Relation(**r) for r in json.loads(rel_path.read_text(encoding="utf-8"))]
            all_relations.extend(rels)
            print(f"    {len(rels)} relations")

    print(f"\nTotal: {len(all_articles)} articles, {len(all_entities)} entities, {len(all_relations)} relations")

    # Build graph
    builder = KnowledgeGraphBuilder()
    graph = builder.build(all_entities, all_relations, all_articles)

    # Save
    save_graph(graph, GRAPH_DIR / "knowledge_graph.json")

    # Print stats
    node_types = {}
    for _, attrs in graph.nodes(data=True):
        t = attrs.get("node_type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    edge_types = {}
    for _, _, attrs in graph.edges(data=True):
        t = attrs.get("relation_type", "unknown")
        edge_types[t] = edge_types.get(t, 0) + 1

    print("\nNode types:")
    for t, count in sorted(node_types.items()):
        print(f"  {t}: {count}")

    print("\nEdge types:")
    for t, count in sorted(edge_types.items()):
        print(f"  {t}: {count}")

    # Cross-version stats
    cross = sum(1 for _, _, a in graph.edges(data=True) if a.get("law_id") == "cross")
    print(f"\nCross-version edges: {cross}")


if __name__ == "__main__":
    main()
