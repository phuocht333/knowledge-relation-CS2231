"""Knowledge Graph visualizer using NetworkX + Pyvis.

Produces lightweight interactive HTML files suitable for:
- Embedding screenshots into slides
- Visualizing query results in-browser
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from src.extraction.models import Entity, Relation

# ── Colour palettes ────────────────────────────────────────────────
ENTITY_TYPE_COLORS: dict[str, str] = {
    "khái_niệm": "#4A90D9",  # blue
    "điều_luật": "#E8913A",  # orange
    "quyền_nghĩa_vụ": "#5BB55B",  # green
    "mức_hưởng": "#9B59B6",  # purple
    "xử_phạt": "#E74C3C",  # red
}

RELATION_TYPE_COLORS: dict[str, str] = {
    "định_nghĩa": "#3498DB",
    "quy_định": "#E67E22",
    "áp_dụng": "#2ECC71",
    "tham_chiếu": "#9B59B6",
    "bao_gồm": "#1ABC9C",
    "điều_kiện": "#F1C40F",
    "hạn_chế": "#E74C3C",
    "liên_quan": "#95A5A6",
}

ENTITY_TYPE_LABELS: dict[str, str] = {
    "khái_niệm": "Khái niệm",
    "điều_luật": "Điều luật",
    "quyền_nghĩa_vụ": "Quyền/Nghĩa vụ",
    "mức_hưởng": "Mức hưởng",
    "xử_phạt": "Xử phạt",
}


# ── Data loading ───────────────────────────────────────────────────


def load_data(
    entities_path: str | Path,
    relations_path: str | Path,
) -> tuple[list[Entity], list[Relation]]:
    """Load entities and relations from JSON files."""
    entities_path = Path(entities_path)
    relations_path = Path(relations_path)

    entities = [
        Entity(**e) for e in json.loads(entities_path.read_text(encoding="utf-8"))
    ]
    relations = [
        Relation(**r) for r in json.loads(relations_path.read_text(encoding="utf-8"))
    ]
    return entities, relations


# ── NetworkX graph builder ─────────────────────────────────────────


def build_nx_graph(
    entities: list[Entity],
    relations: list[Relation],
) -> nx.DiGraph:
    """Build a NetworkX directed graph from entities and relations."""
    g = nx.DiGraph()

    for e in entities:
        g.add_node(
            e.id,
            label=e.name,
            entity_type=(
                e.entity_type.value
                if hasattr(e.entity_type, "value")
                else e.entity_type
            ),
            description=e.description,
            source_article=e.source_article,
        )

    entity_ids = {e.id for e in entities}
    for r in relations:
        if r.source_id in entity_ids and r.target_id in entity_ids:
            g.add_edge(
                r.source_id,
                r.target_id,
                relation_type=(
                    r.relation_type.value
                    if hasattr(r.relation_type, "value")
                    else r.relation_type
                ),
                description=r.description,
                source_article=r.source_article,
            )

    return g


# ── Subgraph filters ──────────────────────────────────────────────


def filter_by_articles(
    g: nx.DiGraph,
    article_numbers: list[int],
) -> nx.DiGraph:
    """Return subgraph containing only nodes from specified articles."""
    nodes = [
        n for n, d in g.nodes(data=True) if d.get("source_article") in article_numbers
    ]
    return g.subgraph(nodes).copy()


def filter_by_keyword(
    g: nx.DiGraph,
    keyword: str,
    hops: int = 1,
) -> nx.DiGraph:
    """Return subgraph around nodes whose name contains the keyword.

    Args:
        g: Full graph.
        keyword: Search term (case-insensitive).
        hops: How many hops of neighbours to include around matched nodes.
    """
    keyword_lower = keyword.lower()
    seed_nodes = {
        n for n, d in g.nodes(data=True) if keyword_lower in d.get("label", "").lower()
    }

    if not seed_nodes:
        print(f"  No nodes found matching '{keyword}'")
        return nx.DiGraph()

    # Expand by hops (undirected for neighbourhood)
    neighbours: set[str] = set(seed_nodes)
    for _ in range(hops):
        new_neighbours: set[str] = set()
        for node in neighbours:
            new_neighbours.update(g.predecessors(node))
            new_neighbours.update(g.successors(node))
        neighbours.update(new_neighbours)

    return g.subgraph(neighbours).copy()


def filter_by_entity_type(
    g: nx.DiGraph,
    entity_types: list[str],
) -> nx.DiGraph:
    """Return subgraph containing only nodes of specified entity types."""
    nodes = [n for n, d in g.nodes(data=True) if d.get("entity_type") in entity_types]
    return g.subgraph(nodes).copy()


# ── Pyvis renderer ─────────────────────────────────────────────────


def _legend_html() -> str:
    """Generate an HTML legend overlay for entity types."""
    items = "".join(
        f'<div style="display:flex;align-items:center;margin:4px 0">'
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'border-radius:50%;background:{color};margin-right:8px"></span>'
        f'<span style="color:#333;font-size:13px">{ENTITY_TYPE_LABELS.get(etype, etype)}</span>'
        f"</div>"
        for etype, color in ENTITY_TYPE_COLORS.items()
    )
    return (
        f'<div style="position:fixed;top:16px;right:16px;background:white;'
        f"padding:12px 16px;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.15);"
        f'z-index:9999;font-family:system-ui,sans-serif">'
        f'<div style="font-weight:600;margin-bottom:6px;color:#222;font-size:14px">Loại thực thể</div>'
        f"{items}</div>"
    )


LAW_ID_COLORS: dict[str, str] = {
    "2024": "#2196F3",
    "2013": "#4CAF50",
}


def _ego_subgraph(g: nx.DiGraph, node_id: str, hops: int) -> nx.DiGraph:
    """Extract ego subgraph via BFS predecessors + successors up to *hops*."""
    visited: set[str] = {node_id}
    frontier: set[str] = {node_id}
    for _ in range(hops):
        next_frontier: set[str] = set()
        for n in frontier:
            next_frontier.update(g.predecessors(n))
            next_frontier.update(g.successors(n))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
    return g.subgraph(visited).copy()


def render_ego_html(
    g: nx.DiGraph,
    node_id: str,
    hops: int = 1,
    height: str = "600px",
) -> str:
    """Render an ego-centric subgraph and return self-contained HTML string.

    Designed for embedding inside ``gr.HTML()`` in a Gradio app.
    """
    sub = _ego_subgraph(g, node_id, hops)

    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#FAFAFA",
        font_color="#333333",
        cdn_resources="in_line",
    )

    net.set_options(
        """
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.008,
          "springLength": 180,
          "springConstant": 0.04,
          "damping": 0.5
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "enabled": true, "iterations": 200 }
      },
      "nodes": {
        "font": { "size": 13, "face": "system-ui, sans-serif" },
        "borderWidth": 2,
        "borderWidthSelected": 3
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "font": { "size": 10, "align": "middle", "color": "#888" },
        "smooth": { "type": "curvedCW", "roundness": 0.15 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """
    )

    degrees = dict(sub.degree())
    max_deg = max(degrees.values()) if degrees else 1

    for nid, data in sub.nodes(data=True):
        ntype = data.get("node_type", "entity")
        law = data.get("law_id", "")
        is_center = nid == node_id

        if ntype == "article":
            num = data.get("name", nid)
            title = data.get("title", "")
            label = f"{num}. {title}" if title else num
            color = LAW_ID_COLORS.get(law, "#999999")
            shape = "box"
            tooltip = (
                f"{label}\n"
                f"Điều luật — LĐĐ {law}\n"
                f"{data.get('chapter', '')} {data.get('chapter_title', '')}"
            )
        elif ntype == "chapter":
            label = f"{data.get('name', nid)} - {data.get('title', '')}"
            color = LAW_ID_COLORS.get(law, "#999999")
            shape = "box"
            tooltip = f"{label}\nChương — LĐĐ {law}"
        else:
            etype = data.get("entity_type", "khái_niệm")
            label = data.get("name", nid)
            color = ENTITY_TYPE_COLORS.get(etype, "#999999")
            shape = "dot"
            desc = data.get("description", "")
            tooltip = (
                f"{label}\n"
                f"{ENTITY_TYPE_LABELS.get(etype, etype)}\n"
                f"Điều {data.get('source_article', '?')} — LĐĐ {law}"
            )
            if desc:
                tooltip += f"\n\n{desc}"

        display_label = label if len(label) <= 45 else label[:42] + "..."
        deg = degrees.get(nid, 1)
        size = 12 + (deg / max_deg) * 28

        if is_center:
            size = max(size, 35)
            border_width = 4
            border_color = "#FF5722"
        else:
            border_width = 2
            border_color = color

        net.add_node(
            nid,
            label=display_label,
            title=tooltip,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": color, "border": "#FF5722"},
            },
            size=size,
            shape=shape,
            borderWidth=border_width,
        )

    for src, tgt, data in sub.edges(data=True):
        rtype = data.get("relation_type", "liên_quan")
        edge_law = data.get("law_id", "")
        if edge_law == "cross":
            color = "#FF9800"
        else:
            color = RELATION_TYPE_COLORS.get(rtype, "#CCCCCC")
        desc = data.get("description", "")
        tooltip = f"{rtype}\n{desc}" if desc else rtype
        net.add_edge(src, tgt, title=tooltip, color=color, label=rtype)

    html = net.generate_html()
    legend = _legend_html()
    html = html.replace("</body>", legend + "\n</body>")
    # Wrap in iframe via base64 data URI so gr.HTML() can render the full document
    b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
    return (
        f'<iframe src="data:text/html;base64,{b64}" '
        f'width="100%" height="{height}" '
        f'style="border:none;"></iframe>'
    )


def render(
    g: nx.DiGraph,
    output_path: str | Path = "graph.html",
    title: str = "Knowledge Graph — Luật Đất đai 2024",
    height: str = "900px",
    width: str = "100%",
    bg_color: str = "#FAFAFA",
    show_legend: bool = True,
) -> Path:
    """Render a NetworkX graph to an interactive Pyvis HTML file.

    Args:
        g: The NetworkX graph to render.
        output_path: Where to write the HTML file.
        title: Title displayed at the top of the page.
        height: CSS height of the canvas.
        width: CSS width of the canvas.
        bg_color: Background colour.
        show_legend: Whether to overlay a colour legend.

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor=bg_color,
        font_color="#333333",
        heading=title,
        cdn_resources="in_line",  # self-contained HTML, no external CDN needed
    )

    # Physics settings – stable layout, not too bouncy
    net.set_options(
        """
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.008,
          "springLength": 180,
          "springConstant": 0.04,
          "damping": 0.5
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 200
        }
      },
      "nodes": {
        "font": { "size": 13, "face": "system-ui, sans-serif" },
        "borderWidth": 2,
        "borderWidthSelected": 3
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "font": { "size": 10, "align": "middle", "color": "#888" },
        "smooth": { "type": "curvedCW", "roundness": 0.15 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """
    )

    # Compute degree for sizing
    degrees = dict(g.degree())
    max_deg = max(degrees.values()) if degrees else 1

    # Add nodes
    for node_id, data in g.nodes(data=True):
        etype = data.get("entity_type", "khái_niệm")
        color = ENTITY_TYPE_COLORS.get(etype, "#999999")
        deg = degrees.get(node_id, 1)
        size = 12 + (deg / max_deg) * 28  # 12–40px range

        label = data.get("label", node_id)
        # Truncate long labels for display
        display_label = label if len(label) <= 40 else label[:37] + "..."

        tooltip = (
            f"<b>{label}</b><br>"
            f"<i>{ENTITY_TYPE_LABELS.get(etype, etype)}</i><br>"
            f"Điều {data.get('source_article', '?')}<br><br>"
            f"{data.get('description', '')}"
        )

        net.add_node(
            node_id,
            label=display_label,
            title=tooltip,
            color=color,
            size=size,
            shape="dot",
        )

    # Add edges
    for src, tgt, data in g.edges(data=True):
        rtype = data.get("relation_type", "liên_quan")
        color = RELATION_TYPE_COLORS.get(rtype, "#CCCCCC")

        tooltip = f"<b>{rtype}</b><br>" f"{data.get('description', '')}"

        net.add_edge(
            src,
            tgt,
            title=tooltip,
            color=color,
            label=rtype,
        )

    net.save_graph(str(output_path))

    # Inject legend HTML if requested
    if show_legend:
        html = output_path.read_text(encoding="utf-8")
        html = html.replace("</body>", _legend_html() + "\n</body>")
        output_path.write_text(html, encoding="utf-8")

    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()
    print(f"  ✓ Saved {output_path} ({node_count} nodes, {edge_count} edges)")
    return output_path
