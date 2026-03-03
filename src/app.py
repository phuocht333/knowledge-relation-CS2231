import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import networkx as nx

from src.config import EMBEDDINGS_DIR, GRAPH_DIR, ARTICLES_DIR
from src.embedding.text_embedder import TextEmbedder
from src.embedding.vector_store import VectorStore
from src.graph.kg_store import load_graph
from src.retrieval.retriever import HybridRetriever
from src.qa.agent import LegalQAAgent
from src.visualization.graph_visualizer import (
    ENTITY_TYPE_LABELS,
    render_ego_html,
)


def load_system():
    print("Loading system components...")
    graph = load_graph(GRAPH_DIR / "knowledge_graph.json")
    vector_store = VectorStore.load(EMBEDDINGS_DIR)
    embedder = TextEmbedder()
    retriever = HybridRetriever(vector_store, graph, embedder)
    agent = LegalQAAgent(retriever)
    return agent, graph


def get_graph_stats(graph: nx.DiGraph) -> str:
    # Count by law version
    law_nodes = {"2013": 0, "2024": 0, "other": 0}
    node_types = {}
    for _, attrs in graph.nodes(data=True):
        t = attrs.get("node_type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
        law = attrs.get("law_id", "other")
        if law in law_nodes:
            law_nodes[law] += 1
        else:
            law_nodes["other"] += 1

    edge_types = {}
    cross_edges = 0
    for _, _, attrs in graph.edges(data=True):
        t = attrs.get("relation_type", "unknown")
        edge_types[t] = edge_types.get(t, 0) + 1
        if attrs.get("law_id") == "cross":
            cross_edges += 1

    stats = f"## Knowledge Graph Statistics\n\n"
    stats += f"**Total nodes:** {graph.number_of_nodes()}\n\n"
    stats += f"**Total edges:** {graph.number_of_edges()}\n\n"
    stats += f"**Cross-version edges:** {cross_edges}\n\n"

    stats += "### Nodes by Law Version\n"
    stats += f"- LĐĐ 2013: {law_nodes['2013']}\n"
    stats += f"- LĐĐ 2024: {law_nodes['2024']}\n"
    stats += f"- Other: {law_nodes['other']}\n"

    stats += "\n### Node Types\n"
    for t, count in sorted(node_types.items()):
        stats += f"- {t}: {count}\n"

    stats += "\n### Edge Types\n"
    for t, count in sorted(edge_types.items()):
        stats += f"- {t}: {count}\n"

    return stats


def _build_node_choices(graph: nx.DiGraph) -> list[tuple[str, str]]:
    """Build (display_label, node_id) pairs for dropdown."""
    choices: list[tuple[str, str]] = []
    for nid, data in graph.nodes(data=True):
        ntype = data.get("node_type", "entity")
        law = data.get("law_id", "")
        if ntype == "article":
            label = f"{data.get('name', nid)} - {data.get('title', '')} (LĐĐ {law})"
        elif ntype == "chapter":
            label = f"{data.get('name', nid)} - {data.get('title', '')} (LĐĐ {law})"
        else:
            etype = data.get("entity_type", "")
            etype_label = ENTITY_TYPE_LABELS.get(etype, etype)
            label = f"{data.get('name', nid)} ({etype_label}, LĐĐ {law})"
        choices.append((label, nid))
    choices.sort(key=lambda x: x[0])
    return choices


def _node_info_md(graph: nx.DiGraph, node_id: str) -> str:
    """Build a markdown summary for a single node."""
    data = graph.nodes[node_id]
    ntype = data.get("node_type", "entity")
    law = data.get("law_id", "")
    in_deg = graph.in_degree(node_id)
    out_deg = graph.out_degree(node_id)

    lines = [f"### {data.get('name', node_id)}"]

    if ntype == "article":
        title = data.get("title", "")
        if title:
            lines.append(f"**Tiêu đề:** {title}")
        lines.append(f"**Loại:** Điều luật")
        lines.append(f"**Luật:** LĐĐ {law}")
        chapter = data.get("chapter", "")
        if chapter:
            lines.append(f"**Chương:** {chapter} — {data.get('chapter_title', '')}")
    elif ntype == "chapter":
        lines.append(f"**Loại:** Chương")
        lines.append(f"**Luật:** LĐĐ {law}")
    else:
        etype = data.get("entity_type", "")
        etype_label = ENTITY_TYPE_LABELS.get(etype, etype)
        lines.append(f"**Loại:** {etype_label}")
        lines.append(f"**Luật:** LĐĐ {law}")
        src_art = data.get("source_article")
        if src_art:
            lines.append(f"**Điều nguồn:** Điều {src_art}")
        desc = data.get("description", "")
        if desc:
            lines.append(f"\n> {desc}")

    lines.append(f"\n**Bậc vào / ra:** {in_deg} / {out_deg}")
    return "\n\n".join(lines)


def create_app():
    agent, graph = load_system()

    def chat_fn(message, history):
        if not message.strip():
            return ""
        result = agent.answer(message)
        return result["answer"]

    def stats_fn():
        return get_graph_stats(graph)

    node_choices = _build_node_choices(graph)

    def explore_fn(selected_node_id: str, hops: int):
        if not selected_node_id:
            return "Vui lòng chọn một node.", "<p>Chưa có dữ liệu.</p>"
        if selected_node_id not in graph:
            return f"Node `{selected_node_id}` không tồn tại.", ""
        info = _node_info_md(graph, selected_node_id)
        html = render_ego_html(graph, selected_node_id, hops=hops)
        return info, html

    examples = [
        # Single law questions
        "Nhà nước thu hồi đất trong trường hợp nào? (LĐĐ 2024)",
        "Bản đồ địa chính là gì?",
        "Quyền của người sử dụng đất theo LĐĐ 2024?",
        "Điều kiện chuyển nhượng quyền sử dụng đất?",
        # Comparison questions
        "So sánh quy định thu hồi đất giữa LĐĐ 2013 và LĐĐ 2024",
        "Điểm mới của Luật Đất đai 2024 so với 2013 về quyền sử dụng đất?",
        "Phân loại đất thay đổi như thế nào giữa 2 phiên bản luật?",
        "So sánh quy hoạch sử dụng đất giữa LĐĐ 2013 và 2024",
    ]

    with gr.Blocks(
        title="Graph RAG - Luật Đất đai 2013 & 2024",
    ) as app:
        gr.Markdown(
            "# Graph RAG - Hỏi đáp & So sánh Luật Đất đai\n"
            "Hệ thống hỏi đáp sử dụng đồ thị tri thức (Knowledge Graph) "
            "kết hợp AI để trả lời câu hỏi và **so sánh** giữa "
            "**Luật Đất đai 2013** (45/2013/QH13) và **Luật Đất đai 2024** (31/2024/QH15)."
        )

        with gr.Tabs():
            with gr.TabItem("Hỏi đáp & So sánh"):
                chatbot = gr.ChatInterface(
                    fn=chat_fn,
                    examples=examples,
                    api_name=False,
                )

            with gr.TabItem("Thống kê đồ thị"):
                stats_btn = gr.Button("Hiển thị thống kê")
                stats_output = gr.Markdown()
                stats_btn.click(fn=stats_fn, outputs=stats_output, api_name=False)

            with gr.TabItem("Khám phá đồ thị"):
                with gr.Row():
                    node_dropdown = gr.Dropdown(
                        choices=node_choices,
                        label="Chọn node",
                        filterable=True,
                        scale=4,
                    )
                    hops_slider = gr.Slider(
                        minimum=1,
                        maximum=3,
                        step=1,
                        value=1,
                        label="Số bước (hops)",
                        scale=1,
                    )
                explore_btn = gr.Button("Hiển thị", variant="primary")

                node_info = gr.Markdown(label="Thông tin node")
                graph_html = gr.HTML(label="Đồ thị")

                explore_btn.click(
                    fn=explore_fn,
                    inputs=[node_dropdown, hops_slider],
                    outputs=[node_info, graph_html],
                    api_name=False,
                )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
