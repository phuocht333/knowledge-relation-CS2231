"""
Merge NotebookLM extraction outputs into entities and relations files.

Expected file structure:
  data/entities/2024_chapter_1.json ... 2024_chapter_6.json
  data/entities/2013_chapter_1.json ... 2013_chapter_14.json

Usage:
  .venv/bin/python scripts/merge_notebooklm.py
  Then:
  .venv/bin/python scripts/03_build_graph.py
  .venv/bin/python scripts/04_embed.py
"""
import json
import re
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.config import ENTITIES_DIR

VALID_ENTITY_TYPES = {"khái_niệm", "điều_luật", "quyền_nghĩa_vụ", "mức_hưởng", "xử_phạt"}
VALID_RELATION_TYPES = {"định_nghĩa", "quy_định", "áp_dụng", "tham_chiếu", "bao_gồm", "điều_kiện", "hạn_chế", "liên_quan"}

LAW_CHAPTERS = {
    "2024": 6,
    "2013": 14,
}


def clean_json_text(text: str) -> str:
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return text


def process_law(law_id: str, num_chapters: int):
    all_entities = []
    all_relations = []
    seen_entity_ids = set()

    for i in range(1, num_chapters + 1):
        path = ENTITIES_DIR / f"{law_id}_chapter_{i}.json"
        if not path.exists():
            print(f"  Warning: {path.name} not found, skipping")
            continue

        raw = path.read_text(encoding="utf-8")
        cleaned = clean_json_text(raw)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"  Error parsing {path.name}: {e}")
            continue

        entities = data.get("entities", [])
        relations = data.get("relations", [])

        valid_count = 0
        for ent in entities:
            eid = ent.get("id", "")
            if not eid or not ent.get("name"):
                continue
            # Prefix entity IDs with law_id to avoid collisions
            if not eid.startswith(f"{law_id}_"):
                eid = f"{law_id}_{eid}"
                ent["id"] = eid
            if ent.get("entity_type", "") not in VALID_ENTITY_TYPES:
                ent["entity_type"] = "khái_niệm"

            if eid not in seen_entity_ids:
                seen_entity_ids.add(eid)
                all_entities.append({
                    "id": eid,
                    "name": ent.get("name", ""),
                    "entity_type": ent.get("entity_type", "khái_niệm"),
                    "description": ent.get("description", ""),
                    "source_article": ent.get("source_article", 0),
                    "source_text": ent.get("source_text", ""),
                    "law_id": law_id,
                })
                valid_count += 1

        rel_count = 0
        for rel in relations:
            src = rel.get("source_id", "")
            tgt = rel.get("target_id", "")
            if not src or not tgt:
                continue
            # Prefix relation IDs
            if not src.startswith(f"{law_id}_"):
                src = f"{law_id}_{src}"
            if not tgt.startswith(f"{law_id}_"):
                tgt = f"{law_id}_{tgt}"
            if src not in seen_entity_ids or tgt not in seen_entity_ids:
                continue
            if rel.get("relation_type", "") not in VALID_RELATION_TYPES:
                rel["relation_type"] = "liên_quan"

            all_relations.append({
                "source_id": src,
                "target_id": tgt,
                "relation_type": rel.get("relation_type", "liên_quan"),
                "description": rel.get("description", ""),
                "source_article": rel.get("source_article", 0),
                "law_id": law_id,
            })
            rel_count += 1

        print(f"  {law_id}_chapter_{i}: {valid_count} entities, {rel_count} relations")

    return all_entities, all_relations


def main():
    print("=== Merging NotebookLM outputs ===\n")

    for law_id, num_chapters in LAW_CHAPTERS.items():
        print(f"Processing LĐĐ {law_id} ({num_chapters} chapters):")
        entities, relations = process_law(law_id, num_chapters)

        if entities:
            ent_path = ENTITIES_DIR / f"entities_{law_id}.json"
            ent_path.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  -> Saved {len(entities)} entities to {ent_path.name}")

        if relations:
            rel_path = ENTITIES_DIR / f"relations_{law_id}.json"
            rel_path.write_text(json.dumps(relations, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  -> Saved {len(relations)} relations to {rel_path.name}")

        if not entities:
            print(f"  -> No chapter files found for LĐĐ {law_id}")
        print()

    print("Next steps:")
    print("  .venv/bin/python scripts/03_build_graph.py")
    print("  .venv/bin/python scripts/04_embed.py")
    print("  .venv/bin/python -m src.app")


if __name__ == "__main__":
    main()
