"""Visualize the extracted knowledge graph.

Usage examples:
    # Full overview (sampled to keep it readable)
    python scripts/05_visualize.py

    # Specific articles
    python scripts/05_visualize.py --articles 78 79 80

    # Search by keyword (with 1-hop neighbours)
    python scripts/05_visualize.py --keyword "thu hồi đất"

    # Search with 2-hop expansion
    python scripts/05_visualize.py --keyword "bồi thường" --hops 2

    # Filter by entity type
    python scripts/05_visualize.py --types khái_niệm quyền_nghĩa_vụ
"""

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from pathlib import Path

from src.config import ENTITIES_DIR, DATA_DIR
from src.visualization.graph_visualizer import (
    load_data,
    build_nx_graph,
    filter_by_articles,
    filter_by_keyword,
    filter_by_entity_type,
    render,
)

OUTPUT_DIR = DATA_DIR / "visualizations"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize knowledge graph")
    parser.add_argument("--law", default="2024", help="Law version (default: 2024)")
    parser.add_argument(
        "--articles",
        type=int,
        nargs="+",
        help="Filter by article numbers (e.g. --articles 78 79 80)",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        help="Filter by keyword search (e.g. --keyword 'thu hồi đất')",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=1,
        help="Number of hops for keyword search (default: 1)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["khái_niệm", "điều_luật", "quyền_nghĩa_vụ", "mức_hưởng", "xử_phạt"],
        help="Filter by entity types",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200,
        help="Max nodes for full graph view (default: 200)",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────
    entities_path = ENTITIES_DIR / f"entities_{args.law}.json"
    relations_path = ENTITIES_DIR / f"relations_{args.law}.json"

    # Fallback to backup directory
    if not entities_path.exists():
        entities_path = ENTITIES_DIR / "backup" / f"entities_{args.law}.json"
        relations_path = ENTITIES_DIR / "backup" / f"relations_{args.law}.json"

    if not entities_path.exists():
        print(f"Error: Cannot find entities file for law {args.law}")
        sys.exit(1)

    print(f"=== Visualizing Knowledge Graph — Luật Đất đai {args.law} ===")
    entities, relations = load_data(entities_path, relations_path)
    print(f"  Loaded {len(entities)} entities, {len(relations)} relations")

    g = build_nx_graph(entities, relations)

    # ── Apply filters ─────────────────────────────────────────
    title = f"Knowledge Graph — Luật Đất đai {args.law}"
    filename = f"graph_{args.law}"

    if args.articles:
        g = filter_by_articles(g, args.articles)
        art_str = ", ".join(str(a) for a in args.articles)
        title = f"Điều {art_str} — LĐĐ {args.law}"
        filename = f"graph_{args.law}_art{'_'.join(str(a) for a in args.articles)}"
        print(f"  Filtered by articles: {art_str}")

    elif args.keyword:
        g = filter_by_keyword(g, args.keyword, hops=args.hops)
        title = f"'{args.keyword}' ({args.hops}-hop) — LĐĐ {args.law}"
        safe_kw = args.keyword.replace(" ", "_")[:30]
        filename = f"graph_{args.law}_{safe_kw}"
        print(f"  Filtered by keyword: '{args.keyword}' ({args.hops} hops)")

    elif args.types:
        g = filter_by_entity_type(g, args.types)
        title = f"Types: {', '.join(args.types)} — LĐĐ {args.law}"
        filename = f"graph_{args.law}_types"
        print(f"  Filtered by types: {args.types}")

    else:
        # Full graph – sample by top-degree nodes to keep it readable
        node_count = g.number_of_nodes()
        if node_count > args.max_nodes:
            degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)
            top_nodes = [n for n, _ in degrees[: args.max_nodes]]
            g = g.subgraph(top_nodes).copy()
            title += f" (top {args.max_nodes} nodes by degree)"
            print(f"  Sampled top {args.max_nodes}/{node_count} nodes by degree")

    if g.number_of_nodes() == 0:
        print("  No nodes to visualize.")
        sys.exit(0)

    # ── Render ────────────────────────────────────────────────
    output_path = OUTPUT_DIR / f"{filename}.html"
    render(g, output_path=output_path, title=title)
    print(f"\n  Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
