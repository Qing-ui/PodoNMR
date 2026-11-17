from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_nmr_sample_network(group_scores: pd.DataFrame, curated_dir: str | Path = "data/curated") -> dict[str, pd.DataFrame]:
    curated = Path(curated_dir)
    if group_scores.empty:
        return {"nodes": pd.DataFrame(), "edges": pd.DataFrame()}
    selected = group_scores.copy().reset_index(drop=True)
    selected.insert(0, "network_rank", range(1, len(selected) + 1))
    selected_ids = set(selected["group_id"].astype(str))
    score_lookup = selected.set_index("group_id")["final_score"].astype(float).to_dict()
    edge_path = curated / "fragment_group_network_export_edges.csv"
    if not edge_path.exists():
        return {"nodes": selected, "edges": pd.DataFrame()}
    edges = pd.read_csv(edge_path)
    if edges.empty:
        return {"nodes": selected, "edges": pd.DataFrame()}
    edges["source_group_id"] = edges["source_group_id"].astype(str)
    edges["target_group_id"] = edges["target_group_id"].astype(str)
    edges = edges[edges["source_group_id"].isin(selected_ids) & edges["target_group_id"].isin(selected_ids)].copy()
    if edges.empty:
        return {"nodes": selected, "edges": edges.reset_index(drop=True)}
    edges["source_final_score"] = edges["source_group_id"].map(score_lookup).fillna(0.0)
    edges["target_final_score"] = edges["target_group_id"].map(score_lookup).fillna(0.0)
    edge_strength = pd.to_numeric(edges.get("interpretable_score", 0.0), errors="coerce").fillna(0.0)
    edges["sample_edge_score"] = edge_strength * ((edges["source_final_score"] + edges["target_final_score"]) / 2.0)
    edges = edges.sort_values(["sample_edge_score", "interpretable_score", "source_group_id", "target_group_id"], ascending=[False, False, True, True])
    return {"nodes": selected, "edges": edges.reset_index(drop=True)}


def plot_nmr_sample_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    out_svg: str | Path | None = None,
    out_png: str | Path | None = None,
) -> dict[str, Path]:
    outputs = {}
    if out_svg is None and out_png is None:
        return outputs
    import networkx as nx

    from nmr_fragment_network.network.network_plot import _draw_graph

    graph = nx.Graph()
    for item in nodes.to_dict("records"):
        group_id = str(item["group_id"])
        graph.add_node(group_id, **item)
    for item in edges.to_dict("records"):
        source = str(item["source_group_id"])
        target = str(item["target_group_id"])
        if source in graph and target in graph:
            graph.add_edge(source, target, **item)
    if out_svg is not None:
        _draw_graph(graph, out_svg, size_column="final_score", color_by="quality")
        outputs["svg"] = Path(out_svg)
    if out_png is not None:
        _draw_graph(graph, out_png, size_column="final_score", color_by="quality")
        outputs["png"] = Path(out_png)
    return outputs


def _network_scores(prior_scores: dict[str, float], edges: pd.DataFrame, network_weight: float) -> dict[str, float]:
    scores = {group_id: 0.0 for group_id in prior_scores}
    if edges.empty or network_weight <= 0:
        return scores
    max_score = max(float(edges["interpretable_score"].max()), 1.0) if "interpretable_score" in edges else 1.0
    for item in edges.to_dict("records"):
        source = str(item.get("source_group_id", ""))
        target = str(item.get("target_group_id", ""))
        edge = float(item.get("interpretable_score", 0.0)) / max_score
        if source in scores and target in prior_scores:
            scores[source] += prior_scores[target] * edge * network_weight
        if target in scores and source in prior_scores:
            scores[target] += prior_scores[source] * edge * network_weight
    return scores


def _attach_group_quality(result: pd.DataFrame, curated: Path) -> pd.DataFrame:
    audit_path = curated / "fragment_group_node_quality_audit.csv"
    if not audit_path.exists() or result.empty:
        return result
    audit = pd.read_csv(audit_path)
    columns = ["group_id", "group_members", "quality_flags", "dominant_skeleton", "strongest_partners", "example_compounds"]
    return result.merge(audit[[column for column in columns if column in audit.columns]], on="group_id", how="left")


def _attach_class_similarity_support(classes: pd.DataFrame, similar: pd.DataFrame) -> pd.DataFrame:
    if classes.empty or similar.empty or "skeleton_type" not in similar.columns:
        return classes
    rows = []
    for skeleton_type, group in similar.groupby("skeleton_type", dropna=False):
        rows.append(
            {
                "skeleton_type": str(skeleton_type),
                "top_similar_compound_count": int(len(group)),
                "best_nmr_similarity": float(pd.to_numeric(group["nmr_similarity"], errors="coerce").max())
                if "nmr_similarity" in group.columns
                else 0.0,
                "example_similar_compounds": ";".join(group["compound_id"].astype(str).head(5)) if "compound_id" in group.columns else "",
            }
        )
    support = pd.DataFrame(rows)
    result = classes.merge(support, on="skeleton_type", how="left")
    result["top_similar_compound_count"] = result["top_similar_compound_count"].fillna(0).astype(int)
    result["best_nmr_similarity"] = result["best_nmr_similarity"].fillna(0.0)
    result["example_similar_compounds"] = result["example_similar_compounds"].fillna("")
    return result
