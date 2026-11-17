from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from nmr_fragment_network.nmr_rules.features import NMR_FEATURE_COLUMNS, build_sample_feature_row

FULL_NMR_COLUMNS = list(NMR_FEATURE_COLUMNS)
STRUCTURE_INPUT_FIELDS = {
    'smiles', 'canonical_smiles', 'skeleton_type', 'compound_id', 'fragment_label',
    'group_label', 'true_label', 'true_skeleton', 'true_group', 'graph_group',
    'structure', 'inchi', 'inchikey', 'molfile', 'group_id', 'fragment_id',
}


def retrieval_indices(train: np.ndarray, test: np.ndarray, scaler: StandardScaler, top: int = 10):
    """NMR feature-space reference search. Smaller distance means closer NMR descriptors.

    This is NOT exact peak assignment or a validated exact-structure identification.
    """
    T = scaler.transform(train)
    Q = scaler.transform(test)
    dist = pairwise_distances(Q, T, metric='euclidean')
    order = np.argsort(dist, axis=1, kind='stable')[:, :top]
    d_sorted = np.take_along_axis(dist, order, axis=1)
    return order, d_sorted


def validate_nmr_input(sample: pd.DataFrame) -> pd.DataFrame:
    """Validate the original combined 13C / linked 1H / carbon type input.

    Each measured carbon is a stable carbon_id, repeated for separately reported
    protons attached to the same carbon (e.g., CH2). 13C/C_type must agree
    between all rows of that carbon. A Cq or carbonyl carbon has empty delta_H.
    SMILES or target scaffold are never accepted at prediction time.
    """
    if sample is None or sample.empty:
        raise ValueError('NMR input is empty')
    cols = {str(c).strip().lower() for c in sample.columns}
    forbidden = cols & STRUCTURE_INPUT_FIELDS
    if forbidden:
        raise ValueError(f'Unknown input contains structure labels: {sorted(forbidden)}')
    for col in ('carbon_id','delta_C','delta_H','C_type'):
        if col not in sample.columns:
            raise ValueError(f'Required original NMR input column missing: {col}')
    df = sample.copy()
    if 'sample_id' in df and df.sample_id.dropna().astype(str).str.strip().nunique() > 1:
        raise ValueError('Provide one sample per CSV file.')
    df['carbon_id'] = df.carbon_id.astype('string').str.strip()
    if df.carbon_id.isna().any() or df.carbon_id.str.lower().isin(['', 'nan', 'none']).any():
        raise ValueError('Empty carbon_id: distinct carbons require distinct IDs')
    df['delta_C'] = pd.to_numeric(df['delta_C'], errors='coerce')
    proton_text = df['delta_H'].astype('string').str.strip()
    df['delta_H'] = pd.to_numeric(df['delta_H'], errors='coerce')
    if (proton_text.notna() & proton_text.ne('') & df.delta_H.isna()).any():
        raise ValueError('Enter a numeric 1H shift in ppm or leave the field blank.')
    if not np.isfinite(df.delta_C).all() or not df.delta_C.between(-20,300).all():
        raise ValueError('Invalid 13C shift')
    if df.delta_H.notna().any() and (not np.isfinite(df.delta_H.dropna()).all() or not df.delta_H.dropna().between(-1,15).all()):
        raise ValueError('Invalid 1H shift')
    df['C_type'] = df['C_type'].fillna('').astype(str).str.strip()
    allowed = {'CH3','CH2','CH','Cq','C=O','COOR','CHO'}
    if (~df.C_type.isin(allowed)).any():
        raise ValueError('C_type must be assigned experimentally (CH3/CH2/CH/Cq/C=O/COOR/CHO)')
    for cid, rows in df.groupby('carbon_id'):
        if float(rows.delta_C.max()-rows.delta_C.min())>.01:
            raise ValueError(f'Inconsistent delta_C for carbon {cid}')
        if rows.C_type.nunique()!=1:
            raise ValueError(f'Inconsistent carbon type for carbon {cid}')
    if df.carbon_id.nunique()<8:
        raise ValueError('At least eight carbon resonance IDs required')
    if df.delta_H.notna().sum()==0:
        raise ValueError('Full NMR model requires 1H shifts paired with carbons; do not substitute 13C-only input')
    return df


def input_features(sample: pd.DataFrame) -> pd.DataFrame:
    df = validate_nmr_input(sample)
    feat = build_sample_feature_row(df)
    return pd.DataFrame([[float(feat[k]) for k in FULL_NMR_COLUMNS]], columns=FULL_NMR_COLUMNS)
