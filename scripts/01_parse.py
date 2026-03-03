import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.config import LAWS, ARTICLES_DIR
from src.parsing.article_parser import parse_articles


def main():
    print("=== Step 1: Parsing articles (both law versions) ===")

    all_articles = []
    for law_id, law_info in LAWS.items():
        source = law_info["source_text"]
        if not source.exists():
            print(f"  Warning: {source} not found, skipping {law_info['name']}")
            continue

        raw_text = source.read_text(encoding="utf-8")
        articles = parse_articles(raw_text, law_id=law_id)
        print(f"\n{law_info['name']}: {len(articles)} articles")

        for art in articles[:3]:
            print(f"  Điều {art.article_number}: {art.title} ({art.chapter})")
        if len(articles) > 3:
            print(f"  ... and {len(articles) - 3} more")

        # Save per-law file
        output_path = ARTICLES_DIR / f"articles_{law_id}.json"
        data = [art.model_dump() for art in articles]
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved to {output_path}")

        all_articles.extend(articles)

    # Save combined file
    combined_path = ARTICLES_DIR / "articles_all.json"
    combined_data = [art.model_dump() for art in all_articles]
    combined_path.write_text(json.dumps(combined_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal: {len(all_articles)} articles saved to {combined_path}")


if __name__ == "__main__":
    main()
