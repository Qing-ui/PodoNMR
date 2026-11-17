from __future__ import annotations

from collections import Counter

import pandas as pd

from nmr_fragment_network.normalize import infer_c_type, parse_float


CARBON_BINS = list(range(0, 231, 10))
PROTON_BINS = [x / 2 for x in range(0, 25)]
CANONICAL_TYPES = ["CH3", "CH2", "CH", "Cq", "C=O", "COOR", "CHO"]

NMR_FEATURE_COLUMNS = (
    [
        "n_carbons",
        "n_hsqc",
        "n_carbonyl",
        "n_ketone_region",
        "n_sp2",
        "n_oxygenated",
        "n_terminal_methylene_like",
        "frac_hsqc",
        "mean_delta_C",
        "std_delta_C",
        "mean_delta_H",
        "std_delta_H",
    ]
    + [f"type_{name}" for name in CANONICAL_TYPES]
    + [f"cbin_{lo}_{lo + 10}" for lo in CARBON_BINS[:-1]]
    + [f"hbin_{lo:.1f}_{lo + 0.5:.1f}" for lo in PROTON_BINS[:-1]]
)


def build_sample_feature_row(sample: pd.DataFrame) -> dict[str, float]:
    """Count each carbon exactly once; include linked 1H shifts without duplicating C."""
    records = [_row_values(row) for row in sample.to_dict("records")]
    carbon_rows = [r for r in records if r["delta_C"] is not None]
    # An actual sample may lack a carbon index; in that case use the combination
    # of chemical shift and carbon type as an approximate chemical resonance key.
    # Curated known samples supply a stable carbon_id from atom-index metadata.
    unique_c = {}
    for r in carbon_rows:
        key = r.get("carbon_id")
        if key is None or str(key) in {"", "nan", "None"}:
            key = (round(r["delta_C"], 4), r["C_type"])
        unique_c.setdefault(str(key), r)
    carbons = list(unique_c.values())
    delta_c = [r["delta_C"] for r in carbons]
    delta_h = [r["delta_H"] for r in carbon_rows if r["delta_H"] is not None]
    protonated_carbon_keys = set()
    for r in carbon_rows:
        if r["delta_H"] is None:
            continue
        key = r.get("carbon_id")
        if key is None or str(key) in {"", "nan", "None"}:
            key = (round(r["delta_C"], 4), r["C_type"])
        protonated_carbon_keys.add(str(key))
    type_counts = Counter(r["C_type"] for r in carbons)
    features = {col: 0.0 for col in NMR_FEATURE_COLUMNS}
    features.update({
        "n_carbons": float(len(carbons)),
        "n_hsqc": float(len(protonated_carbon_keys)),
        "n_carbonyl": float(sum(v >= 165 for v in delta_c)),
        "n_ketone_region": float(sum(190 <= v <= 220 for v in delta_c)),
        "n_sp2": float(sum(100 <= v <= 160 for v in delta_c)),
        "n_oxygenated": float(sum(55 <= v <= 90 for v in delta_c)),
        "n_terminal_methylene_like": float(sum(105 <= r["delta_C"] <= 115 and r["C_type"] == "CH2" for r in carbons)),
        "frac_hsqc": float(len(protonated_carbon_keys) / max(len(carbons), 1)),
        "mean_delta_C": _mean(delta_c),
        "std_delta_C": _std(delta_c),
        "mean_delta_H": _mean(delta_h),
        "std_delta_H": _std(delta_h),
    })
    for name in CANONICAL_TYPES:
        features[f"type_{name}"] = float(type_counts.get(name, 0))
    for value in delta_c:
        key = _bin_key("cbin", value, CARBON_BINS, 10.0)
        if key in features:
            features[key] += 1.0
    for value in delta_h:
        key = _bin_key("hbin", value, PROTON_BINS, 0.5)
        if key in features:
            features[key] += 1.0
    return features


def _row_values(row: dict) -> dict:
    delta_h = parse_float(row.get("delta_H"))
    c_type = str(row.get("C_type") or "").strip()
    return {
        "delta_C": parse_float(row.get("delta_C")),
        "delta_H": delta_h,
        "C_type": c_type or infer_c_type(row.get("delta_C"), [delta_h] if delta_h is not None else []),
        "carbon_id": row.get("carbon_id"),
    }


def _bin_key(prefix: str, value: float, bins: list[float], width: float) -> str:
    for lo, hi in zip(bins[:-1], bins[1:]):
        if lo <= value < hi:
            if prefix == "hbin":
                return f"{prefix}_{lo:.1f}_{hi:.1f}"
            return f"{prefix}_{int(lo)}_{int(hi)}"
    lo = bins[-2]
    hi = bins[-1]
    if value >= hi:
        lo = hi - width
    if prefix == "hbin":
        return f"{prefix}_{lo:.1f}_{lo + width:.1f}"
    return f"{prefix}_{int(lo)}_{int(lo + width)}"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5)
