import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.config import ARTICLES_DIR, ENTITIES_DIR, LAWS
from src.parsing.models import Article
from src.extraction.entity_extractor import EntityExtractor

CHECKPOINT_DIR = ENTITIES_DIR / "checkpoints"


def main():
    print("=== Step 2: Extracting entities via Gemini (both law versions) ===")

    extractor = EntityExtractor()

    for law_id in LAWS:
        entities_path = ENTITIES_DIR / f"entities_{law_id}.json"
        if entities_path.exists():
            print(f"  Skipping {LAWS[law_id]['name']}: {entities_path} already exists")
            continue

        articles_path = ARTICLES_DIR / f"articles_{law_id}.json"
        if not articles_path.exists():
            print(f"  Warning: {articles_path} not found, skipping")
            continue

        raw = json.loads(articles_path.read_text(encoding="utf-8"))
        articles = [Article(**a) for a in raw]
        print(f"\n{LAWS[law_id]['name']}: {len(articles)} articles")

        result = extractor.extract_all(
            articles,
            delay=0.5,
            checkpoint_dir=CHECKPOINT_DIR,
            law_id=law_id,
            save_every=10,
        )

        # Tag with law_id
        for e in result.entities:
            e.law_id = law_id
        for r in result.relations:
            r.law_id = law_id

        # Save final output
        entities_path = ENTITIES_DIR / f"entities_{law_id}.json"
        entities_data = [e.model_dump() for e in result.entities]
        entities_path.write_text(
            json.dumps(entities_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        relations_path = ENTITIES_DIR / f"relations_{law_id}.json"
        relations_data = [r.model_dump() for r in result.relations]
        relations_path.write_text(
            json.dumps(relations_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(
            f"  Saved {len(result.entities)} entities, {len(result.relations)} relations"
        )


if __name__ == "__main__":
    main()
