from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from nmr_fragment_network.fragment_names import fragment_code, fragment_name


def node_color_for_flags(flags: str) -> str:
    text = str(flags)
    if "network_isolated" in text:
        return "#d73027"
    if "top8_unstable" in text:
        return "#fdae61"
    if "low_support" in text or "rare_node" in text:
        return "#fee08b"
    if "broad_skeleton" in text:
        return "#91bfdb"
    if "background_excluded" in text:
        return "#bdbdbd"
    return "#1a9850"


def _draw_graph(graph: nx.Graph, out: str | Path, size_column: str = "group_support_compounds", color_by: str = "quality") -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8.5))
    fig.subplots_adjust(left=.06, right=.58, top=.92, bottom=.09)
    pos = nx.spring_layout(graph, seed=23, weight="interpretable_score", k=0.8)
    strengths = [float(data.get("interpretable_score", 0.0) or 0.0) for _, _, data in graph.edges(data=True)]
    max_strength = max(strengths) if strengths else 1.0
    edge_widths = [0.8 + 4.0 * strength / max_strength for strength in strengths]
    size_values = [float(data.get(size_column, 0.0) or 0.0) for _, data in graph.nodes(data=True)]
    max_size_value = max(size_values) if size_values else 1.0
    node_sizes = [
        220 + 900 * (value / max_size_value if max_size_value else 0.0)
        for value in size_values
    ]
    node_colors = _node_colors(graph, color_by)
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


def _node_colors(graph: nx.Graph, color_by: str) -> list:
    if color_by == "external_support":
        values = [float(data.get("external_support_compounds", 0.0) or 0.0) for _, data in graph.nodes(data=True)]
        max_value = max(values) if values else 1.0
        cmap = plt.colormaps["YlOrRd"]
        return [cmap(value / max_value if max_value else 0.0) for value in values]
    return [node_color_for_flags(str(data.get("quality_flags", ""))) for _, data in graph.nodes(data=True)]
