"""Train-only purged-cluster OOF selection of eleven bounded shrinkage rules."""
import argparse
from dataclasses import fields, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import run_formal as f
from bisurfgnn import lightweight_controls as light

CANDIDATES=('global_0','global_025','global_05','global_075','global_1',
            'anchor_q25','anchor_q50','anchor_q75','gap_q25','gap_q50','gap_q75')


def cluster_split(frame):
    clusters=sorted(set(frame.heldout_cluster_id)|set(frame.partner_cluster_id),
                    key=lambda x:hashlib.sha256(('41:'+str(x)).encode()).hexdigest())
    if len(clusters)<3:raise ValueError('At least three clusters required')
    mapping={c:i%3 for i,c in enumerate(clusters)}
    assigned=frame.heldout_cluster_id.map(mapping).to_numpy()
    if (frame.assign(inner=assigned).groupby('prediction_group_id').inner.nunique()>1).any():
        raise ValueError('Prediction group split across folds')
    splits=[]
    for k in range(3):
        held={c for c,v in mapping.items() if v==k}
        train=~(frame.heldout_cluster_id.isin(held)|frame.partner_cluster_id.isin(held)).to_numpy()
        val=assigned==k
        if not train.any() or not val.any():raise ValueError('Empty purged inner partition')
        train_clusters=set(frame.loc[train,'heldout_cluster_id'])|set(frame.loc[train,'partner_cluster_id'])
        if train_clusters & held:raise ValueError('Chemical cluster leakage')
        if set(frame.loc[train,'prediction_group_id']) & set(frame.loc[val,'prediction_group_id']):raise ValueError('Group leakage')
        splits.append((train,val,dict(inner=k,train_points=int(train.sum()),validation_points=int(val.sum()),
            excluded_points=int((~train & ~val).sum()),heldout_clusters=sorted(held),
            training_clusters=sorted(train_clusters),cluster_overlap=0,group_overlap=0)))
    return splits,assigned


def select_candidate(scores):
    if set(scores)!=set(CANDIDATES) or not np.isfinite(list(scores.values())).all():raise ValueError('Invalid candidate scores')
    best=min(scores.values())
    return next(name for name in CANDIDATES if scores[name]<=best+1e-5)


def anchors(table):return np.minimum(table.mono_a.numpy(),table.mono_b.numpy()).astype(float)


def rule_parameters(table):
    if table.role!='train':raise ValueError('Rule parameters require training role')
    return dict(anchor=np.maximum(anchors(table),0),gap=np.abs(table.mono_a.numpy().astype(float)-table.mono_b.numpy()))


def quantiles(table):
    return {key:{str(q):max(float(np.quantile(values,q/100)),1e-6) for q in (25,50,75)}
            for key,values in rule_parameters(table).items()}


def apply_rule(name, anchor, gap, ridge, parameters):
    if name not in CANDIDATES:raise ValueError('Unknown rule')
    global_weights=dict(global_0=0.,global_025=.25,global_05=.5,global_075=.75,global_1=1.)
    if name in global_weights:w=global_weights[name]
    else:
        key,qname=name.split('_q');q=parameters[key][qname]
        w=np.maximum(anchor,0)/(np.maximum(anchor,0)+q) if key=='anchor' else q/(gap+q)
    return anchor+w*(ridge-anchor)


def subset(table, mask, role):
    updates={field.name:getattr(table,field.name)[mask] for field in fields(table)
             if isinstance(getattr(table,field.name),(np.ndarray,torch.Tensor))}
    return replace(table,**updates,role=role)


def fit(table,banks):
    state=light.fit_feature_state(table,banks,seed=41)
    x=light.feature_matrix(table,banks,state,'mono_dose_ridge')
    model=light.ridge_fit(x,table,alpha=.01,solver='primal')
    return state,model


def predict(table,banks,state,model):
    return light.ridge_predict(light.feature_matrix(table,banks,state,'mono_dose_ridge'),table,model)


def macro(table,values):return f.learned.surface_macro_rmse(table.target.numpy(),values,table.surface_ids)


def run(output):
    output=Path(output).resolve()
    if not output.is_relative_to(f.ROOT/'runs/audit_revision'):raise ValueError('Invalid output scope')
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    stage=f.ROOT/'runs/audit_revision/staged_screen_v1'
    plan=f.read_json(stage/'plan.json');f.check_seal(plan)
    parent_path=Path(plan['parent_plan']['path']);parent=f.read_json(parent_path);f.check_seal(parent)
    metadata_path=f.ROOT/'data/processed/one_new_scaffold_benchmark_unclipped/group_assignments.parquet'
    metadata=pd.read_parquet(metadata_path)[['prediction_group_id','partner_cluster_id']]
    if metadata.prediction_group_id.duplicated().any():raise ValueError('Duplicate group metadata')
    sources=[f.learned._file_record(path) for path in (Path(__file__).resolve(),
        f.ROOT/'docs/CROSSFIT_SHRINKAGE_PROTOCOL.md',f.ROOT/'tests/test_crossfit_shrinkage.py')]
    f.verify(plan['sources']);sources += plan['sources']
    inputs=[f.learned._file_record(stage/'plan.json'),f.learned._file_record(metadata_path),f.learned._file_record(parent_path)]
    f.seal(output/'protocol.json',dict(sources=sources,candidates=CANDIDATES,
        scope='post_observation_development_train_only_selection',inputs=inputs))
    summaries=[];trace=[str(metadata_path)]
    for fold in range(5):
        print(f'FOLD {fold}: train-only OOF starts',flush=True)
        dest=output/f'fold_{fold}';dest.mkdir()
        train_rec=plan['inputs'][str(fold)]['train'];f.verify([train_rec]);inputs.append(train_rec)
        if Path(train_rec['path']).name!='train.parquet':raise ValueError('Forbidden train path')
        frame=pd.read_parquet(train_rec['path']);trace.append(train_rec['path'])
        frame=frame[frame.primary_metric_mask & frame.response_mask].reset_index(drop=True)
        frame=frame.merge(metadata,on='prediction_group_id',how='left',validate='many_to_one')
        if frame.partner_cluster_id.isna().any():raise ValueError('Missing partner cluster')
        banks=SimpleNamespace(drug_ids=tuple(sorted(set(frame.drug_a_id)|set(frame.drug_b_id))),
                              cell_ids=tuple(sorted(set(frame.cell_id))))
        table=f.learned.build_compact_point_table(frame,banks,role='train',fold=fold,monotherapy_upper_bound=3)
        splits,assignment=cluster_split(frame)
        oof=np.full((len(frame),len(CANDIDATES)),np.nan);coverage=np.zeros(len(frame),dtype=int);split_info=[]
        for mask,valmask,info in splits:
            training=subset(table,mask,'train');valid=subset(table,valmask,'validation')
            features,model=fit(training,banks);ridge=predict(valid,banks,features,model)
            params=quantiles(training);anchor=anchors(valid);gap=np.abs(valid.mono_a.numpy().astype(float)-valid.mono_b.numpy())
            for i,name in enumerate(CANDIDATES):oof[valmask,i]=apply_rule(name,anchor,gap,ridge,params)
            coverage[valmask]+=1;split_info.append(dict(**info,rule_parameters=params))
            print(f'FOLD {fold}: inner {info["inner"]+1}/3 complete ({info["train_points"]} train points)',flush=True)
        if not np.all(coverage==1) or not np.isfinite(oof).all():raise ValueError('Incomplete OOF coverage')
        scores={name:macro(table,oof[:,i]) for i,name in enumerate(CANDIDATES)}
        chosen=select_candidate(scores)
        evidence=pd.DataFrame(dict(source_id=table.source_rows,surface_id=table.surface_ids,
            group_id=table.group_ids,inner=assignment,target=table.target.numpy()))
        for i,name in enumerate(CANDIDATES):evidence[name]=oof[:,i]
        evidence.to_parquet(dest/'oof.parquet',index=False)
        # Seal model selection BEFORE opening outer validation responses.
        selection=f.seal(dest/'selection.json',dict(fold=fold,selected=chosen,oof_scores=scores,splits=split_info,
            oof=f.learned._file_record(dest/'oof.parquet'),selection_role='train_only',outer_validation_loaded=False))
        print(f'FOLD {fold}: choice sealed: {chosen}; now evaluate outer validation',flush=True)
        full_state,full_model=fit(table,banks);params=quantiles(table)
        val_rec=plan['inputs'][str(fold)]['validation'];f.verify([val_rec]);inputs.append(val_rec)
        if Path(val_rec['path']).name!='validation.parquet':raise ValueError('Forbidden validation path')
        valframe=pd.read_parquet(val_rec['path']);trace.append(val_rec['path'])
        valframe=valframe[valframe.primary_metric_mask & valframe.response_mask].reset_index(drop=True)
        # Expand lookup vocabulary only, not fitted coefficients or feature statistics.
        all_banks=SimpleNamespace(drug_ids=tuple(sorted(set(banks.drug_ids)|set(valframe.drug_a_id)|set(valframe.drug_b_id))),
                                  cell_ids=tuple(sorted(set(banks.cell_ids)|set(valframe.cell_id))))
        valid=f.learned.build_compact_point_table(valframe,all_banks,role='validation',fold=fold,monotherapy_upper_bound=3)
        # mono_dose_ridge consumes no entity mappings; retain only the required
        # vocabulary contract, not stale permutation indices from the old ordering.
        prediction_state=dict(drug_ids=list(all_banks.drug_ids),cell_ids=list(all_banks.cell_ids))
        ridge=predict(valid,all_banks,prediction_state,full_model)
        owner=parent if fold==0 else plan;base=parent_path.parent if fold==0 else stage
        job=next(j for j in owner['inventory']['jobs'] if j['track']=='main' and j['model']=='mono_dose_ridge' and j['seed']==41 and j['fold']==fold)
        old_path=base/'jobs'/job['id']/'receipt.json';old=f.read_json(old_path)
        f.verify_receipts({'jobs':[job]},[old],owner['manifest_sha256']);inputs += [f.learned._file_record(old_path),old['prediction']]
        reference=pd.read_parquet(old['prediction']['path']).sort_values('source_id');trace.append(old['prediction']['path'])
        order=np.argsort(valid.source_rows)
        np.testing.assert_array_equal(valid.source_rows[order],reference.source_id)
        np.testing.assert_allclose(ridge[order],reference.prediction,atol=1e-7,rtol=0)
        anchor=anchors(valid);gap=np.abs(valid.mono_a.numpy().astype(float)-valid.mono_b.numpy())
        predictions={name:apply_rule(name,anchor,gap,ridge,params) for name in CANDIDATES}
        out=pd.DataFrame(dict(source_id=valid.source_rows,surface_id=valid.surface_ids,group_id=valid.group_ids,
            cluster_id=valid.cluster_ids,target=valid.target.numpy(),fold=fold,selected=predictions[chosen],**predictions))
        out.to_parquet(dest/'predictions.parquet',index=False)
        f.learned._write_checkpoint_atomic(dest/'model.pt',dict(features=prediction_state,model=full_model,parameters=params,selected=chosen))
        loaded=torch.load(dest/'model.pt',weights_only=True,map_location='cpu')
        np.testing.assert_allclose(ridge,predict(valid,all_banks,loaded['features'],loaded['model']),atol=1e-10,rtol=0)
        np.testing.assert_allclose(out.selected,apply_rule(loaded['selected'],anchor,gap,ridge,loaded['parameters']),atol=1e-10,rtol=0)
        squared=pd.DataFrame({name:(valid.target.numpy().astype(float)-out[name].to_numpy())**2 for name in ['selected',*CANDIDATES]})
        squared['surface_id']=valid.surface_ids
        surface=np.sqrt(squared.groupby('surface_id').mean())
        surface.to_parquet(dest/'surface_errors.parquet')
        metrics={name:macro(valid,values) for name,values in predictions.items()}
        result=f.seal(dest/'result.json',dict(fold=fold,selected=chosen,selected_rmse=metrics[chosen],validation_scores=metrics,
            selection=f.learned._file_record(dest/'selection.json'),checkpoint=f.learned._file_record(dest/'model.pt'),
            prediction=f.learned._file_record(dest/'predictions.parquet'),surface_errors=f.learned._file_record(dest/'surface_errors.parquet'),
            baseline_reproduction_max_abs=float(np.max(abs(ridge[order]-reference.prediction.to_numpy())))))
        summaries.append(result)
    f.verify(sources+inputs)
    f.seal(output/'summary.json',dict(scope='post_observation_development_not_confirmation',folds=summaries,
        mean_selected_rmse=float(np.mean([r['selected_rmse'] for r in summaries])),
        mean_ridge_rmse=float(np.mean([r['validation_scores']['global_1'] for r in summaries])),
        mean_hsa_rmse=float(np.mean([r['validation_scores']['global_0'] for r in summaries])),
        sources=sources,inputs=inputs,read_trace=trace,test_response_files_read=0))
    print('DONE: five-fold crossfit shrinkage experiment; no further tasks launched.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args().output_dir)
