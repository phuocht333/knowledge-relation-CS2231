import json
import re
import time
from pathlib import Path

from google import genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.extraction.models import (
    Entity,
    Relation,
    ExtractionResult,
    EntityType,
    RelationType,
)
from src.extraction.prompts import ENTITY_EXTRACTION_SYSTEM, ENTITY_EXTRACTION_USER
from src.parsing.models import Article


def _build_extraction_schema() -> dict:
    """Build JSON schema with enum constraints for Gemini structured output."""
    entity_type_values = [e.value for e in EntityType]
    relation_type_values = [r.value for r in RelationType]

    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "entity_type": {
                            "type": "string",
                            "enum": entity_type_values,
                        },
                        "description": {"type": "string"},
                        "source_article": {"type": "integer"},
                        "source_text": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "name",
                        "entity_type",
                        "description",
                        "source_article",
                        "source_text",
                    ],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "target_id": {"type": "string"},
                        "relation_type": {
                            "type": "string",
                            "enum": relation_type_values,
                        },
                        "description": {"type": "string"},
                        "source_article": {"type": "integer"},
                    },
                    "required": [
                        "source_id",
                        "target_id",
                        "relation_type",
                        "description",
                        "source_article",
                    ],
                },
            },
        },
        "required": ["entities", "relations"],
    }


EXTRACTION_SCHEMA = _build_extraction_schema()


class EntityExtractor:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def extract_from_article(
        self, article: Article, max_retries: int = 3
    ) -> ExtractionResult:
        prompt = ENTITY_EXTRACTION_USER.format(
            article_header=article.summary_header,
            article_content=article.content[:3000],
            article_number=article.article_number,
        )

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=ENTITY_EXTRACTION_SYSTEM,
                        response_mime_type="application/json",
                        response_schema=EXTRACTION_SCHEMA,
                    ),
                )
                break
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    wait = 30 * (attempt + 1)
                    print(
                        f"    Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})..."
                    )
                    time.sleep(wait)
                else:
                    print(f"    API error: {e}")
                    if attempt == max_retries - 1:
                        return ExtractionResult(entities=[], relations=[])
                    time.sleep(5)
        else:
            return ExtractionResult(entities=[], relations=[])

        text = response.text.strip()
        # Remove markdown code blocks if present
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"  Warning: Failed to parse JSON for Điều {article.article_number}")
            return ExtractionResult(entities=[], relations=[])

        entities = []
        for e in data.get("entities", []):
            try:
                entities.append(
                    Entity(
                        id=e["id"],
                        name=e["name"],
                        entity_type=EntityType(e["entity_type"]),
                        description=e["description"],
                        source_article=e["source_article"],
                        source_text=e.get("source_text", ""),
                    )
                )
            except (ValueError, KeyError) as err:
                print(f"  Warning: Skipping entity {e.get('id', '?')}: {err}")

        relations = []
        entity_ids = {e.id for e in entities}
        for r in data.get("relations", []):
            try:
                if r["source_id"] in entity_ids and r["target_id"] in entity_ids:
                    relations.append(
                        Relation(
                            source_id=r["source_id"],
                            target_id=r["target_id"],
                            relation_type=RelationType(r["relation_type"]),
                            description=r["description"],
                            source_article=r["source_article"],
                        )
                    )
            except (ValueError, KeyError) as err:
                print(f"  Warning: Skipping relation: {err}")

        return ExtractionResult(entities=entities, relations=relations)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoint_path(checkpoint_dir: Path, law_id: str) -> Path:
        return checkpoint_dir / f"checkpoint_{law_id}.json"

    @staticmethod
    def _load_checkpoint(
        checkpoint_file: Path,
    ) -> tuple[set[int], list[Entity], list[Relation]]:
        """Load a previous checkpoint. Returns (processed_articles, entities, relations)."""
        if not checkpoint_file.exists():
            return set(), [], []

        try:
            data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            processed = set(data.get("processed_articles", []))
            entities = [Entity(**e) for e in data.get("entities", [])]
            relations = [Relation(**r) for r in data.get("relations", [])]
            print(
                f"  ✓ Loaded checkpoint: {len(processed)} articles processed, "
                f"{len(entities)} entities, {len(relations)} relations"
            )
            return processed, entities, relations
        except (json.JSONDecodeError, Exception) as err:
            print(f"  Warning: Failed to load checkpoint ({err}), starting fresh")
            return set(), [], []

    @staticmethod
    def _save_checkpoint(
        checkpoint_file: Path,
        processed_articles: set[int],
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Persist current progress to a checkpoint file."""
        data = {
            "processed_articles": sorted(processed_articles),
            "entities": [e.model_dump() for e in entities],
            "relations": [r.model_dump() for r in relations],
        }
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _delete_checkpoint(checkpoint_file: Path) -> None:
        """Remove checkpoint file after successful completion."""
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("  ✓ Checkpoint cleaned up")

    # ------------------------------------------------------------------
    # Main extraction loop
    # ------------------------------------------------------------------

    def extract_all(
        self,
        articles: list[Article],
        delay: float = 1.0,
        checkpoint_dir: Path | None = None,
        law_id: str = "2024",
        save_every: int = 10,
    ) -> ExtractionResult:
        """Extract entities/relations from all articles with checkpoint support.

        Args:
            articles: List of articles to process.
            delay: Seconds to wait between API calls.
            checkpoint_dir: Directory to store checkpoint files. None = no checkpointing.
            law_id: Identifier used for the checkpoint filename.
            save_every: Save a checkpoint every N articles.
        """
        all_entities: list[Entity] = []
        all_relations: list[Relation] = []
        processed_articles: set[int] = set()
        checkpoint_file: Path | None = None

        # Load existing checkpoint
        if checkpoint_dir is not None:
            checkpoint_file = self._checkpoint_path(checkpoint_dir, law_id)
            processed_articles, all_entities, all_relations = self._load_checkpoint(
                checkpoint_file
            )

        total = len(articles)
        skipped = 0

        for i, article in enumerate(articles):
            # Skip already-processed articles
            if article.article_number in processed_articles:
                skipped += 1
                continue

            print(
                f"  [{i+1}/{total}] Extracting Điều {article.article_number}: {article.title}"
            )
            result = self.extract_from_article(article)
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)
            processed_articles.add(article.article_number)
            print(
                f"    -> {len(result.entities)} entities, {len(result.relations)} relations"
            )

            # Save checkpoint every N newly-processed articles
            newly_processed = len(processed_articles) - skipped  # only count this run
            if (
                checkpoint_file is not None
                and newly_processed > 0
                and (len(processed_articles) % save_every == 0)
            ):
                self._save_checkpoint(
                    checkpoint_file, processed_articles, all_entities, all_relations
                )
                print(
                    f"  ✓ Checkpoint saved ({len(processed_articles)}/{total} articles)"
                )

            if i < total - 1:
                time.sleep(delay)

        if skipped > 0:
            print(f"  Skipped {skipped} already-processed articles (from checkpoint)")

        # Final checkpoint save to capture any remaining articles
        if checkpoint_file is not None and len(processed_articles) % save_every != 0:
            self._save_checkpoint(
                checkpoint_file, processed_articles, all_entities, all_relations
            )

        # Deduplicate entities by name (keep first occurrence)
        seen_names: dict[str, str] = {}
        unique_entities = []
        id_mapping: dict[str, str] = {}

        for e in all_entities:
            normalized = e.name.lower().strip()
            if normalized in seen_names:
                id_mapping[e.id] = seen_names[normalized]
            else:
                seen_names[normalized] = e.id
                unique_entities.append(e)
                id_mapping[e.id] = e.id

        # Remap relation IDs and deduplicate
        unique_relations = []
        seen_rels = set()
        for r in all_relations:
            src = id_mapping.get(r.source_id, r.source_id)
            tgt = id_mapping.get(r.target_id, r.target_id)
            key = (src, tgt, r.relation_type)
            if key not in seen_rels:
                seen_rels.add(key)
                unique_relations.append(
                    Relation(
                        source_id=src,
                        target_id=tgt,
                        relation_type=r.relation_type,
                        description=r.description,
                        source_article=r.source_article,
                    )
                )

        print(
            f"\nTotal: {len(unique_entities)} unique entities, {len(unique_relations)} unique relations"
        )

        # Clean up checkpoint after successful completion
        if checkpoint_file is not None:
            self._delete_checkpoint(checkpoint_file)

        return ExtractionResult(entities=unique_entities, relations=unique_relations)
