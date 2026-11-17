from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from nmr_fragment_network.fragment_names import fragment_code, fragment_name


def _draw_graph(graph: nx.Graph, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.subplots_adjust(left=.06, right=.58, top=.92, bottom=.09)
    pos = nx.spring_layout(graph, seed=23, weight="interpretable_score", k=0.8)
    strengths = [float(data.get("interpretable_score", 0.0) or 0.0) for _, _, data in graph.edges(data=True)]
    max_strength = max(strengths) if strengths else 1.0
    edge_widths = [0.8 + 4.0 * strength / max_strength for strength in strengths]
    size_values = [float(data["final_score"]) for _, data in graph.nodes(data=True)]
    max_size_value = max(size_values) if size_values else 1.0
    node_sizes = [
        220 + 900 * (value / max_size_value if max_size_value else 0.0)
        for value in size_values
    ]
    node_colors = ["#1a9850" if score >= 0.5 else "#8b95a1" for score in size_values]
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color="#777777", alpha=0.55, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, edgecolors="#222222", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels={node:fragment_code(node) for node in graph.nodes},
                            font_size=10, font_weight="bold", ax=ax)
    ax.axis("off")
    ax.set_title("Fragment co-occurrence network", fontsize=13)
    from textwrap import fill
    pairs=sorted(((fragment_code(node), fragment_name(node)) for node in graph.nodes),key=lambda x:x[0])
    for j,(num,name) in enumerate(pairs):
        yy=.91-j*(.86/max(9,len(pairs)))
        fig.text(.60,yy,num,fontsize=9,ha="left",va="top",weight="bold",color="#263e4c")
        fig.text(.645,yy,fill(name,55),fontsize=8.5,ha="left",va="top",color="#263e4c")
    fig.savefig(out, dpi=220, bbox_inches="tight",facecolor="white")
    plt.close(fig)
