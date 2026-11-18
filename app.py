from __future__ import annotations

from base64 import b64encode
import sys
import json
from tempfile import TemporaryDirectory
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from html import escape
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nmr_fragment_network.ui_support25 import recommend_nmr_network
from nmr_fragment_network.fragment_names import fragment_name, fragment_label, fragment_code


DATA = ROOT / "data"
MODEL = ROOT / "data" / "support25" / "support25_integrated_model.joblib"
DISPLAY_SETTINGS = DATA / "display_settings.json"
REQUIRED_COLUMNS = {"carbon_id", "delta_C", "delta_H", "C_type"}
NODE_COUNT = 10
REFERENCE_COUNT = 10
CLASS_COUNT = 8
MOTIF_COLUMNS = ["official_model_rank", "fragment_no", "fragment_name_en", "role",
                 "group_probability", "predicted_at_0_5"]
TABLE_LABELS = {
    "class_rank": "Rank",
    "skeleton_type": "Scaffold",
    "class_score": "Scaffold score",
    "official_model_rank": "Rank",
    "fragment_no": "Motif ID",
    "fragment_name_en": "Structural motif",
    "role": "Scale",
    "group_probability": "Motif score",
    "predicted_at_0_5": "Score ≥ 0.50",
    "compound_id": "Reference ID",
    "nmr_similarity": "NMR similarity",
    "nmr_distance": "NMR distance",
}
SKELETON_LABELS = {
    "贝壳杉/扁枝杉烷型": "Kaurane/phyllocladane",
    "松香烷型": "Abietane",
    "海松烷型": "Pimarane",
    "二萜内酯型": "Diterpene lactones",
    "桃拓烷型": "Totarane",
    "玫瑰烷型": "Rosane",
    "环氧型半日花烷型": "Epoxy-labdane",
    "罗汉松型": "Podocarpane",
}


st.set_page_config(page_title="PodoNMR", layout="wide")
st.markdown(
    """
    <style>
    #MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"], [data-testid="stHeaderActionElements"] {display: none !important;}
    header[data-testid="stHeader"] {height: 0 !important; visibility: hidden !important;}
    .block-container {padding-top: 1.15rem; max-width: 1480px;}
    section[data-testid="stSidebar"] {background: #f6f8fb; border-right: 1px solid #dfe6ee;}
    h1 {font-size: 2.05rem; margin-bottom: .2rem; color: #17202a;}
    h2, h3 {letter-spacing: 0;}
    .callout {
        border: 1px solid #cfdce5;
        border-left: 5px solid #2b6f73;
        background: #f8fbfb;
        padding: 14px 16px;
        color: #1f2d36;
        border-radius: 6px;
        margin: 10px 0 14px 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e7ee;
        border-radius: 8px;
        padding: 12px 14px;
    }
    div[data-testid="stMetricValue"] {font-size: 1.42rem;}
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 700;
        background: #2f6f73 !important;
        border-color: #2f6f73 !important;
        color: #ffffff !important;
    }
    .stButton > button:hover {
        background: #245a5e !important;
        border-color: #245a5e !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def sample_id(data: pd.DataFrame) -> str:
    if "sample_id" in data.columns and not data["sample_id"].dropna().empty:
        value = str(data["sample_id"].dropna().iloc[0]).strip()
        if value:
            return value
    return "unknown"


def english_skeleton(value: object) -> str:
    text = str(value)
    return SKELETON_LABELS.get(text, text)


def english_display(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ("skeleton_type", "dominant_skeleton", "best_skeleton"):
        if column in data.columns:
            data[column] = data[column].map(english_skeleton)
    if {"name", "fragment_name_en"} <= set(data.columns):
        data["name"] = data["fragment_name_en"]
    return data


def load_input(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    return pd.read_csv(upload, dtype={"carbon_id": "string", "sample_id": "string"})


def save_input(data: pd.DataFrame, run_dir: Path) -> Path:
    input_path = run_dir / "input.csv"
    data.to_csv(input_path, index=False, encoding="utf-8-sig")
    return input_path


def run_analysis(data: pd.DataFrame, node_count: int, compound_count: int, class_count: int) -> dict:
    temp_dir = TemporaryDirectory(prefix="podonmr-")
    run_dir = Path(temp_dir.name)
    input_path = save_input(data, run_dir)
    paths = {
        "groups": run_dir / "group_recommendations.csv",
        "motif_scores": run_dir / "motif_scores.csv",
        "similar": run_dir / "similar_compounds.csv",
        "classes": run_dir / "class_recommendations.csv",
        "network_nodes": run_dir / "network_nodes.csv",
        "network_edges": run_dir / "network_edges.csv",
        "network_svg": run_dir / "network.svg",
        "network_png": run_dir / "network.png",
    }
    result = recommend_nmr_network(
        input_path,
        DATA,
        MODEL,
        out_network_svg=paths["network_svg"],
        out_network_png=paths["network_png"],
        top_groups=node_count,
        top_compounds=compound_count,
        top_classes=class_count,
    )
    for key in ("groups", "motif_scores", "similar", "classes", "network_nodes", "network_edges"):
        english_display(result[key]).to_csv(paths[key], index=False, encoding="utf-8-sig")
    return {"result": result, "paths": paths, "run_dir": run_dir, "temp_dir": temp_dir, "input": data.copy()}


def distinct_descriptor_references(similar: pd.DataFrame) -> pd.DataFrame:
    if similar.empty or "nmr_similarity" not in similar.columns:
        return similar
    values = pd.to_numeric(similar["nmr_similarity"], errors="coerce").fillna(0.0)
    return similar[values < 0.9999].reset_index(drop=True)


def domain_call(nearest_similarity: float, class_score: float, thresholds: dict[str, float]) -> str:
    if nearest_similarity >= thresholds["q25"] and class_score >= 0.5:
        return "in-domain"
    if nearest_similarity >= thresholds["q10"] or class_score >= 0.5:
        return "borderline"
    return "outside"


def write_run_report(
    run_dir: Path,
    result: dict[str, pd.DataFrame],
    top_class: str,
    best_compound: str,
    best_similarity: float,
    nearest_reference: str,
    nearest_reference_similarity: float | None,
    domain_status: str,
    thresholds: dict[str, float],
) -> dict[str, Path]:
    summary_rows = [
        {"item": "top_class", "value": top_class},
        {"item": "best_compound", "value": best_compound},
        {"item": "best_similarity", "value": round(best_similarity, 6)},
        {"item": "nearest_distinct_descriptor_reference", "value": nearest_reference},
        {"item": "nearest_distinct_descriptor_similarity", "value": (
            round(nearest_reference_similarity, 6) if nearest_reference_similarity is not None else ""
        )},
        {"item": "reference_domain", "value": domain_status},
        {"item": "motifs_at_or_above_0_50", "value": int(result["motif_scores"]["predicted_at_0_5"].sum())},
        {"item": "network_edges", "value": len(result["network_edges"])},
        {"item": "class_candidates", "value": len(result["classes"])},
        {"item": "similar_compounds", "value": len(result["similar"])},
        {"item": "reference_similarity_q10", "value": thresholds["q10"]},
        {"item": "reference_similarity_q25", "value": thresholds["q25"]},
        {"item": "reference_similarity_median", "value": thresholds["median"]},
    ]
    summary = pd.DataFrame(summary_rows)
    summary_path = run_dir / "analysis_summary.csv"
    report_path = run_dir / "analysis_report.html"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    html = _analysis_report_html(summary, result, run_dir / "network.png")
    report_path.write_text(html, encoding="utf-8")
    return {"analysis_summary": summary_path, "analysis_report": report_path}


def _analysis_report_html(summary: pd.DataFrame, result: dict[str, pd.DataFrame], network_path: Path) -> str:
    values = dict(zip(summary["item"].astype(str), summary["value"].astype(str)))
    network_image = b64encode(network_path.read_bytes()).decode("ascii")
    cards = [
        ("Top class", values.get("top_class", "")),
        ("Reference domain", values.get("reference_domain", "")),
        ("Nearest reference", f"{values['best_compound']} ({values['best_similarity']})"),
        ("Motifs scoring ≥ 0.50", values["motifs_at_or_above_0_50"]),
        ("Network edges", values.get("network_edges", "")),
        ("Class entries", values.get("class_candidates", "")),
    ]
    card_html = "\n".join(f"<div class='card'><span>{escape(label)}</span><b>{escape(value)}</b></div>" for label, value in cards)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PodoNMR Analysis Report</title>
<style>
body {{font-family: Arial, sans-serif; margin: 32px; color: #1f2937;}}
h1 {{font-size: 28px; margin-bottom: 4px;}}
h2 {{font-size: 18px; margin-top: 28px;}}
.subtitle {{color: #64748b; margin-bottom: 20px;}}
.cards {{display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 18px 0 24px;}}
.card {{border: 1px solid #dce3ea; border-radius: 8px; padding: 12px; background: #fbfcfd;}}
.card span {{display: block; font-size: 12px; color: #64748b; margin-bottom: 8px;}}
.card b {{font-size: 17px;}}
table {{border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 20px;}}
th, td {{border: 1px solid #e2e8f0; padding: 7px 8px; text-align: left;}}
th {{background: #f1f5f9;}}
img {{max-width: 100%; border: 1px solid #e2e8f0; border-radius: 8px;}}
</style>
</head>
<body>
<h1>PodoNMR Analysis Report</h1>
<p class="subtitle">Diterpene scaffold and motif candidates from assigned NMR shifts. Model scores are uncalibrated. Use the results to prioritize candidates for structural review.</p>
<div class="cards">{card_html}</div>
<h2>Sample Network</h2>
<img src="data:image/png;base64,{network_image}" alt="Structural motif co-occurrence network">
<h2>Scaffold Ranking</h2>
{result["classes"].to_html(index=False, escape=True)}
<h2>Structural Motif Scores</h2>
{result["motif_scores"][MOTIF_COLUMNS].rename(columns=TABLE_LABELS).to_html(index=False, escape=True)}
<h2>Network Edges</h2>
{result["network_edges"].to_html(index=False, escape=True)}
<h2>Similar Compounds</h2>
{result["similar"].to_html(index=False, escape=True)}
<h2>Analysis Summary</h2>
{summary.to_html(index=False, escape=True)}
</body>
</html>"""


def display_frame(data: pd.DataFrame, columns: list[str], height: int = 340) -> None:
    present = [column for column in columns if column in data.columns]
    frame = data[present] if present else data
    st.dataframe(frame, height=height, width="stretch", column_config=TABLE_LABELS)


def class_probability_figure(classes: pd.DataFrame) -> go.Figure:
    frame = classes.copy().sort_values("class_score", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["class_score"],
            y=frame["skeleton_type"],
            orientation="h",
            marker=dict(color="#2f6f73"),
            hovertemplate="%{y}<br>score=%{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=8, r=18, t=8, b=24),
        xaxis=dict(range=[0, max(1.0, float(frame["class_score"].max()) if not frame.empty else 1.0)], title="score"),
        yaxis=dict(title=""),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1f2937", size=12),
    )
    return fig


def node_score_figure(groups: pd.DataFrame) -> go.Figure:
    frame = groups.copy().sort_values("official_model_rank", ascending=False)
    frame["display_label"] = frame.group_id.map(fragment_label)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["group_probability"],
            y=frame["display_label"],
            orientation="h",
            name="Motif model score",
            marker=dict(color="#2f6f73"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["network_score"],
            y=frame["display_label"],
            orientation="h",
            name="Network context (display only)",
            marker=dict(color="#c58b2a"),
        )
    )
    fig.update_layout(
        barmode="group",
        height=max(480, 60*len(frame)+100),
        margin=dict(l=390, r=70, t=95, b=60),
        xaxis=dict(title="Score", rangemode="tozero"),
        yaxis=dict(title=""),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", y=1.18, x=0),
        font=dict(color="#1f2937", size=12),
    )
    return fig


def network_figure(nodes: pd.DataFrame, edges: pd.DataFrame) -> go.Figure:
    graph = nx.Graph()
    for row in nodes.to_dict("records"):
        graph.add_node(str(row["group_id"]), **row)
    for row in edges.to_dict("records"):
        source = str(row["source_group_id"])
        target = str(row["target_group_id"])
        if source in graph and target in graph:
            graph.add_edge(source, target, **row)
    if not graph.nodes:
        return go.Figure()
    pos = nx.spring_layout(graph, seed=31, weight="sample_edge_score", k=0.92)
    fig = go.Figure()
    max_edge = max([float(data.get("sample_edge_score", 0.0) or 0.0) for _, _, data in graph.edges(data=True)] or [1.0])
    for source, target, data in graph.edges(data=True):
        width = 1.0 + 5.0 * float(data.get("sample_edge_score", 0.0) or 0.0) / max_edge
        fig.add_trace(
            go.Scatter(
                x=[pos[source][0], pos[target][0]],
                y=[pos[source][1], pos[target][1]],
                mode="lines",
                line=dict(width=width, color="rgba(89, 99, 110, 0.42)"),
                hoverinfo="text",
                text=f"{fragment_label(source)} — {fragment_label(target)}<br>edge={float(data.get('sample_edge_score', 0.0)):.3f}",
                showlegend=False,
            )
        )
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    max_score = max(float(data["final_score"]) for _, data in graph.nodes(data=True)) or 1.0
    for node, data in graph.nodes(data=True):
        score = float(data["final_score"])
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        node_text.append(
            f"<b>{fragment_label(node)}</b><br>Motif score: {score:.3f}"
            f"<br>Most common reference scaffold: {data['dominant_skeleton']}"
        )
        node_size.append(18 + 24 * score / max_score)
        node_color.append("#1a9850" if score >= 0.5 else "#8b95a1")
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[fragment_code(k) for k in graph.nodes],
            textposition="middle center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color="#17202a")),
            textfont=dict(size=12, color="#111827"),
            showlegend=False,
        )
    )
    fig.update_layout(
        height=620,
        margin=dict(l=6, r=6, t=8, b=8),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig


settings = json.loads(DISPLAY_SETTINGS.read_text(encoding="utf-8"))
thresholds = settings["reference_thresholds"]


with st.sidebar:
    st.subheader("Sample")
    uploaded = st.file_uploader("Upload sample table", type=["csv"])
    st.caption("Required CSV columns are carbon_id, delta_C, delta_H and C_type. Enter shifts in ppm and leave delta_H blank for carbons with no attached proton.")

try:
    data = load_input(uploaded)
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

st.title("PodoNMR")
st.write("Rank diterpene scaffolds and structural motifs from assigned ¹³C and ¹H chemical shifts and carbon types.")

if data.empty:
    st.info("Upload a sample table to begin.")
    st.stop()

missing = REQUIRED_COLUMNS - set(data.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

sid = sample_id(data)
input_cols = st.columns(4)
input_cols[0].metric("Sample", sid)
input_cols[1].metric("Carbon sites", data["carbon_id"].nunique(dropna=True))
input_cols[2].metric("Carbon types", data["C_type"].nunique(dropna=True))
input_cols[3].metric("Assigned proton shifts", data["delta_H"].notna().sum())

with st.expander("Input table", expanded=False):
    display_frame(data, ["carbon_id", "delta_C", "delta_H", "C_type", "mult", "J"], height=250)

if st.button("Run analysis", type="primary"):
    with st.spinner("Analyzing experimental NMR data..."):
        try:
            st.session_state["analysis_support25"] = run_analysis(data, NODE_COUNT, REFERENCE_COUNT, CLASS_COUNT)
        except ValueError as error:
            st.session_state.pop("analysis_support25", None)
            st.error(str(error))
            st.stop()

analysis = st.session_state.get("analysis_support25")
if not analysis or not analysis["input"].equals(data):
    st.stop()

result = analysis["result"]
paths = analysis["paths"]
classes = english_display(result["classes"])
groups = english_display(result["groups"])
motifs = result["motif_scores"]
edges = english_display(result["network_edges"])
similar = english_display(result["similar"])
result_display = {
    "classes": classes,
    "motif_scores": motifs,
    "network_edges": edges,
    "similar": similar,
}

top_class = classes.iloc[0]["skeleton_type"] if not classes.empty else ""
best_compound = similar.iloc[0]["compound_id"] if not similar.empty else ""
best_similarity = float(similar.iloc[0]["nmr_similarity"]) if not similar.empty else 0.0
reference_similar = distinct_descriptor_references(similar)
nearest_reference = reference_similar.iloc[0]["compound_id"] if not reference_similar.empty else ""
nearest_reference_similarity = float(reference_similar.iloc[0]["nmr_similarity"]) if not reference_similar.empty else None
top_class_score = float(classes.iloc[0]["class_score"]) if not classes.empty else 0.0
domain_status = (
    domain_call(nearest_reference_similarity, top_class_score, thresholds)
    if nearest_reference_similarity is not None else "insufficient reference support"
)
report_paths = write_run_report(
    analysis["run_dir"],
    result_display,
    top_class,
    best_compound,
    best_similarity,
    nearest_reference,
    nearest_reference_similarity,
    domain_status,
    thresholds,
)
paths.update(report_paths)
st.markdown(
    f'<div class="callout"><b>Top scaffold:</b> {top_class} &nbsp; | &nbsp; '
    f'<b>Nearest reference:</b> {best_compound} ({best_similarity:.3f}) &nbsp; | &nbsp; '
    f'<b>Reference domain:</b> {domain_status}</div>',
    unsafe_allow_html=True,
)

summary = st.columns(5)
summary[0].metric("Motifs scoring ≥ 0.50", int(motifs["predicted_at_0_5"].sum()))
summary[1].metric("Network edges", len(edges))
summary[2].metric("Scaffold classes", len(classes))
summary[3].metric("References", len(similar))
summary[4].metric("Reference domain", domain_status)

tab_report, tab_network, tab_tables, tab_files = st.tabs(["Overview", "Network", "Data", "Export"])

with tab_report:
    left, right = st.columns([0.9, 1.1])
    with left:
        st.subheader("Scaffold ranking")
        st.plotly_chart(class_probability_figure(classes), width="stretch")
    with right:
        st.subheader("Top structural motifs")
        if not groups["predicted_at_0_5"].any():
            st.caption("All motif scores fall below 0.50. The chart shows the five highest scores.")
        st.plotly_chart(node_score_figure(groups), width="stretch")
    st.caption("Scaffold and motif scores are uncalibrated. Use these rankings to prioritize candidates for structural review. The network shows co-occurrence among known reference motifs.")
    st.subheader("References with distinct NMR descriptors")
    display_frame(
        reference_similar,
        ["compound_id", "nmr_similarity", "nmr_distance", "compound_name", "skeleton_type", "source"],
        height=220,
    )

with tab_network:
    st.plotly_chart(network_figure(groups, edges), width="stretch")
    st.caption("Structural motif labels: " + "; ".join(fragment_label(k) for k in groups.group_id))
    display_frame(
        edges,
        ["source_group_id", "target_group_id", "interpretable_score", "source_final_score", "target_final_score", "sample_edge_score", "best_skeleton"],
        height=260,
    )

with tab_tables:
    st.subheader("Scaffold ranking")
    display_frame(
        classes,
        ["class_rank", "skeleton_type", "class_score", "top_similar_compound_count", "best_nmr_similarity", "example_similar_compounds"],
        height=240,
    )
    st.subheader("Reference compounds")
    display_frame(
        similar,
        ["compound_id", "nmr_similarity", "nmr_distance", "compound_name", "skeleton_type", "source", "reference"],
        height=310,
    )
    st.subheader("All 28 structural motif scores")
    display_frame(motifs, MOTIF_COLUMNS, height=420)

with tab_files:
    cols = st.columns(2)
    if paths.get("analysis_report") and paths["analysis_report"].exists():
        cols[0].download_button(
            "Download HTML report",
            data=paths["analysis_report"].read_bytes(),
            file_name="analysis_report.html",
            mime="text/html",
        )
    if paths.get("analysis_summary") and paths["analysis_summary"].exists():
        cols[1].download_button(
            "Download summary CSV",
            data=paths["analysis_summary"].read_bytes(),
            file_name="analysis_summary.csv",
            mime="text/csv",
        )

    for key, label, filename, mimetype in [
        ("motif_scores", "Download all motif scores CSV", "motif_scores.csv", "text/csv"),
        ("groups", "Download displayed motifs CSV", "group_recommendations.csv", "text/csv"),
        ("similar", "Download similar compounds CSV", "similar_compounds.csv", "text/csv"),
        ("classes", "Download class ranking CSV", "class_recommendations.csv", "text/csv"),
        ("network_nodes", "Download network nodes CSV", "network_nodes.csv", "text/csv"),
        ("network_edges", "Download network edges CSV", "network_edges.csv", "text/csv"),
        ("network_png", "Download network PNG", "network.png", "image/png"),
        ("network_svg", "Download network SVG", "network.svg", "image/svg+xml"),
    ]:
        if key in paths and paths[key].exists():
            st.download_button(label, data=paths[key].read_bytes(), file_name=filename, mime=mimetype, key=f"export_{key}")
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, "w", compression=ZIP_DEFLATED) as archive:
        for file in sorted(analysis["run_dir"].iterdir()):
            if file.is_file():
                archive.write(file, arcname=file.name)
    st.download_button("Download all results (ZIP)", data=zip_buffer.getvalue(), file_name="PodoNMR-analysis.zip", mime="application/zip")
