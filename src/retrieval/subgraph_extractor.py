import networkx as nx


def format_subgraph_context(
    graph: nx.DiGraph,
    node_ids: list[str],
    articles_content: dict[str, str],
) -> str:
    """Format subgraph as structured text. articles_content keys: '{law_id}_{article_number}'"""
    sections_2024 = []
    sections_2013 = []
    entity_infos = []
    cross_relations = []

    # Collect article numbers and entity info
    article_keys = set()

    for nid in node_ids:
        if not graph.has_node(nid):
            continue
        attrs = graph.nodes[nid]
        node_type = attrs.get("node_type", "")
        law_id = attrs.get("law_id", "2024")

        if node_type == "article":
            # Parse law_id and article number from node id
            parts = nid.split("_article_")
            if len(parts) == 2:
                article_keys.add((parts[0], int(parts[1])))
        elif node_type == "entity":
            entity_infos.append(attrs)
            article_keys.add((law_id, attrs.get("source_article", 0)))

    # Format article content grouped by law version
    for law_id, art_num in sorted(article_keys):
        key = f"{law_id}_{art_num}"
        if key not in articles_content:
            continue
        art_node = f"{law_id}_article_{art_num}"
        if not graph.has_node(art_node):
            continue
        attrs = graph.nodes[art_node]
        title = attrs.get("title", "")
        chapter = attrs.get("chapter", "")
        law_label = "LĐĐ 2024" if law_id == "2024" else "LĐĐ 2013"

        header = f"### Điều {art_num}. {title} [{law_label}]"
        if chapter:
            header += f" ({chapter})"

        content = articles_content[key]
        entry = f"{header}\n{content}"

        if law_id == "2024":
            sections_2024.append(entry)
        else:
            sections_2013.append(entry)

    # Build output
    parts = []
    if sections_2024:
        parts.append("## Luật Đất đai 2024 (31/2024/QH15)\n\n" + "\n\n".join(sections_2024))
    if sections_2013:
        parts.append("## Luật Đất đai 2013 (45/2013/QH13)\n\n" + "\n\n".join(sections_2013))

    # Format entity summaries
    if entity_infos:
        entity_section = "\n## Thực thể liên quan:\n"
        for e in entity_infos:
            law_label = f"LĐĐ {e.get('law_id', '2024')}"
            entity_section += f"- **{e.get('name', '')}** ({e.get('entity_type', '')}, {law_label}): {e.get('description', '')}\n"
        parts.append(entity_section)

    # Format cross-version relations
    for nid in node_ids:
        if not graph.has_node(nid):
            continue
        for _, tgt, attrs in graph.out_edges(nid, data=True):
            if tgt in node_ids and attrs.get("law_id") == "cross":
                cross_relations.append(attrs.get("description", ""))

    if cross_relations:
        parts.append("\n## Liên kết giữa 2 phiên bản:\n" + "\n".join(f"- {r}" for r in cross_relations))

    return "\n\n".join(parts)
