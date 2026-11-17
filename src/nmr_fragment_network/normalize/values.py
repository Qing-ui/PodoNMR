from __future__ import annotations

import math
import re
from typing import Iterable

MISSING_VALUES = {"", "/", "nan", "none", "null", "未", "-"}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return "" if text.lower() in MISSING_VALUES else text


def parse_float(value: object) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace("δ", "").replace("–", "-").replace("—", "-")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        val = float(match.group(0))
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def infer_c_type(carbon_value: object, proton_values: Iterable[float] = ()) -> str:
    text = clean_text(carbon_value)
    lower = text.lower()
    protons = list(proton_values)
    # Prefer explicit carbon-type annotation when present. Distinguish the carbon
    # quartet 'q' (CH3) from 1H peak multiplicity text.
    if any(token in lower for token in ["cho", "aldehyde"]):
        return "CHO"
    if any(token in lower for token in ["coor", "cooh"]):
        return "COOR"
    if "c=o" in lower or "carbonyl" in lower:
        return "C=O"
    if re.search(r"\bch3\b|[, ]q\b|q$", lower):
        return "CH3"
    if re.search(r"\bch2\b|[, ]t\b|t$", lower):
        return "CH2"
    if re.search(r"\bch\b|[, ]d\b|d$", lower):
        return "CH"
    if re.search(r"\bcq\b|\bqc\b|[, ]s\b|s$", lower):
        return "Cq"
    delta = parse_float(carbon_value)
    if delta is not None and delta >= 165 and not protons:
        return "C=O"
    if len(protons) >= 2:
        return "CH2"
    if len(protons) == 1:
        return "CH"  # a methyl cannot be inferred from a single 1H peak alone
    return "Cq"
