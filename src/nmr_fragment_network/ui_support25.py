"""Inference and fragment network for the fitted PodoNMR models."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from nmr_fragment_network.integrated.workflow_full import input_features, retrieval_indices, FULL_NMR_COLUMNS
from nmr_fragment_network.network.nmr_observed import (
    _network_scores, build_nmr_sample_network, plot_nmr_sample_network,
    _attach_class_similarity_support,
)


def recommend_nmr_network(
    input_csv: str | Path,
    data_dir: str | Path,
    model_path: str | Path,
    out_groups: str | Path | None = None,
    out_similar: str | Path | None = None,
    out_classes: str | Path | None = None,
    out_network_nodes: str | Path | None = None,
    out_network_edges: str | Path | None = None,
    out_network_svg: str | Path | None = None,
    out_network_png: str | Path | None = None,
    top_groups: int = 10,
    top_compounds: int = 10,
    top_classes: int = 5,
    network_weight: float = 0.35,
) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    network_path = data_dir / 'ui_support25'
    sample = pd.read_csv(input_csv, dtype={'carbon_id': 'string', 'sample_id': 'string'})
    bundle=joblib.load(model_path)
    if bundle.get('version') != 'multiscale_support25_v1':
        raise ValueError('Expected only 25%-support multiscale integrated bundle')
    feat=input_features(sample)
    if list(bundle['feature_columns']) != FULL_NMR_COLUMNS:
        raise ValueError('NMR model feature columns have changed')
    X=feat.to_numpy(float)
    classes_raw=bundle['scaffold_model'].predict_proba(X)[0]
    class_names=list(bundle['scaffold_model'].classes_)
    classes=pd.DataFrame({'skeleton_type':class_names,'class_probability':classes_raw})
    classes['class_score']=classes.class_probability
    classes=classes.sort_values('class_score',ascending=False,kind='stable').head(top_classes).reset_index(drop=True)
    classes.insert(0,'class_rank',np.arange(1,len(classes)+1))
    classes['model_version'] = 'support25_multiscale'
    pred=np.asarray(bundle['fragment_model'].predict_proba(X))[0]
    targets=bundle['fragment_targets']
    if len(pred)!=len(targets):raise ValueError('Fragment target count mismatch')
    info=pd.read_csv(data_dir / 'fragment_definitions.csv')
    definitions=info.set_index('fragment_id')
    names=[definitions.loc[k,'name_cn'] for k in targets]
    roles=[definitions.loc[k,'scale'] for k in targets]
    raw=pd.DataFrame({'group_id':targets,'name':names,'role':roles,
                      'group_probability':pred,'predicted_at_0_5':pred>=0.5})
    raw=raw.sort_values('group_probability',ascending=False,kind='stable').reset_index(drop=True)
    raw['official_model_rank']=np.arange(1,len(raw)+1)
    edge_list=pd.read_csv(network_path/'fragment_group_network_export_edges.csv')
    # Standalone display context only. Never added to official model scores.
    score_lookup=dict(zip(raw.group_id,raw.group_probability))
    network_score=_network_scores(score_lookup,edge_list,network_weight)
    raw['network_score']=raw.group_id.map(network_score).fillna(0.0)
    raw['final_score']=raw.group_probability
    groups=raw.loc[raw.predicted_at_0_5].head(top_groups).copy()
    if groups.empty:groups=raw.head(min(top_groups,5)).copy()
    node_metadata = pd.read_csv(network_path / 'fragment_network_nodes.csv')
    groups = groups.merge(node_metadata, on='group_id', how='left', validate='one_to_one')
    groups=groups.reset_index(drop=True)
    from nmr_fragment_network.fragment_names import fragment_name, fragment_code
    raw['fragment_no'] = raw.group_id.map(fragment_code)
    raw['fragment_name_en'] = raw.group_id.map(fragment_name)
    groups['fragment_no'] = groups.group_id.map(fragment_code)
    groups['fragment_name_en'] = groups.group_id.map(fragment_name)

    # Reproduce scaffold/reference retrieval, with the same fitted scaler and
    # 315-reference features retained from V4.6. No old fragment model runs.
    reference=pd.DataFrame(bundle['reference_catalog'])
    id_to_pos={str(k):i for i,k in enumerate(bundle['reference_ids'])}
    index=[id_to_pos[str(k)] for k in reference.compound_id]
    ref_array=np.asarray(bundle['reference_feature_matrix'])[index]
    order,distances=retrieval_indices(ref_array,X,bundle['scaler'],top=min(int(top_compounds),len(reference)))
    similar=reference.iloc[order[0]].copy().reset_index(drop=True)
    similar.insert(0,'rank',np.arange(1,len(similar)+1))
    similar['nmr_distance']=distances[0]
    similar['nmr_similarity']=1.0/(1.0+similar.nmr_distance)
    classes=_attach_class_similarity_support(classes,similar)
    sample_network=build_nmr_sample_network(groups,network_path)
    for key in ('nodes',):
        if not sample_network[key].empty:
            sample_network[key]['fragment_no']=sample_network[key].group_id.map(fragment_code)
            sample_network[key]['fragment_name_en']=sample_network[key].group_id.map(fragment_name)
    sample_id=str(sample['sample_id'].dropna().iloc[0]) if 'sample_id' in sample.columns and sample['sample_id'].notna().any() else 'unknown'
    frames={'groups':groups,'motif_scores':raw,'similar':similar,'classes':classes,
            'network_nodes':sample_network['nodes'],'network_edges':sample_network['edges']}
    for frame in frames.values():
        if not frame.empty:frame.insert(0,'sample_id',sample_id)
    paths={'groups':out_groups,'similar':out_similar,'classes':out_classes,
           'network_nodes':out_network_nodes,'network_edges':out_network_edges}
    for key,path in paths.items():
        if path is not None:
            p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
            frames[key].to_csv(p,index=False,encoding='utf-8-sig')
    if out_network_svg or out_network_png:
        plot_nmr_sample_network(frames['network_nodes'],frames['network_edges'],out_network_svg,out_network_png)
    return frames
