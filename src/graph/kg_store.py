import json
from pathlib import Path

import networkx as nx


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    data = {
        "nodes": [],
        "edges": [],
    }
    for node_id, attrs in graph.nodes(data=True):
        data["nodes"].append({"id": node_id, **attrs})
    for src, tgt, attrs in graph.edges(data=True):
        data["edges"].append({"source": src, "target": tgt, **attrs})

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Graph saved to {path}")


def load_graph(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = nx.DiGraph()

    for node in data["nodes"]:
        node_id = node.pop("id")
        graph.add_node(node_id, **node)

    for edge in data["edges"]:
        src = edge.pop("source")
        tgt = edge.pop("target")
        graph.add_edge(src, tgt, **edge)

    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph
