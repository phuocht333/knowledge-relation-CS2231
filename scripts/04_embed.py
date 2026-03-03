import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.config import ARTICLES_DIR, ENTITIES_DIR, EMBEDDINGS_DIR, EMBEDDING_DIM, LAWS
from src.parsing.models import Article
from src.extraction.models import Entity, Relation
from src.embedding.text_embedder import TextEmbedder
from src.embedding.vector_store import VectorStore


def main():
    print("=== Step 4: Computing embeddings (both laws) ===")

    all_articles = []
    all_entities = []
    all_relations = []

    for law_id in LAWS:
        art_path = ARTICLES_DIR / f"articles_{law_id}.json"
        if art_path.exists():
            all_articles.extend([Article(**a) for a in json.loads(art_path.read_text(encoding="utf-8"))])

        ent_path = ENTITIES_DIR / f"entities_{law_id}.json"
        if ent_path.exists():
            all_entities.extend([Entity(**e) for e in json.loads(ent_path.read_text(encoding="utf-8"))])

        rel_path = ENTITIES_DIR / f"relations_{law_id}.json"
        if rel_path.exists():
            all_relations.extend([Relation(**r) for r in json.loads(rel_path.read_text(encoding="utf-8"))])

    embedder = TextEmbedder()
    store = VectorStore(dim=EMBEDDING_DIM)

    # Embed articles
    print(f"\nEmbedding {len(all_articles)} articles...")
    article_texts = [
        f"Điều {a.article_number}. {a.title} (LĐĐ {a.law_id})\n{a.content[:1000]}"
        for a in all_articles
    ]
    article_embeddings = embedder.embed(article_texts)
    article_metadata = [
        {
            "type": "article",
            "id": f"{a.law_id}_article_{a.article_number}",
            "article_number": a.article_number,
            "title": a.title,
            "chapter": a.chapter,
            "law_id": a.law_id,
            "text": article_texts[i][:500],
        }
        for i, a in enumerate(all_articles)
    ]
    store.add(article_embeddings, article_metadata)

    # Embed entities
    print(f"\nEmbedding {len(all_entities)} entities...")
    if all_entities:
        entity_texts = [e.embedding_text for e in all_entities]
        entity_embeddings = embedder.embed(entity_texts)
        entity_metadata = [
            {
                "type": "entity",
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type.value,
                "source_article": e.source_article,
                "law_id": e.law_id,
                "text": e.embedding_text[:500],
            }
            for e in all_entities
        ]
        store.add(entity_embeddings, entity_metadata)

    # Embed relations
    print(f"\nEmbedding {len(all_relations)} relations...")
    if all_relations:
        relation_texts = [r.description for r in all_relations]
        relation_embeddings = embedder.embed(relation_texts)
        relation_metadata = [
            {
                "type": "relation",
                "id": f"rel_{i}",
                "source_id": r.source_id,
                "target_id": r.target_id,
                "relation_type": r.relation_type.value,
                "source_article": r.source_article,
                "law_id": r.law_id,
                "text": r.description[:500],
            }
            for i, r in enumerate(all_relations)
        ]
        store.add(relation_embeddings, relation_metadata)

    # Save
    store.save(EMBEDDINGS_DIR)
    print(f"\nTotal vectors: {store.index.ntotal}")


if __name__ == "__main__":
    main()
