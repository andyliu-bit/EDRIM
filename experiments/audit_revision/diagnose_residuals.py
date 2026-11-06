"""Read-only development residual diagnosis; no model fitting/test responses."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import run_formal as f

MODELS=('mono_dose_ridge','independent_pooled_gnn','bixt__concat','bixt__gate','cross_attention__gate')
BINS=dict(response=[-np.inf,0,.25,.5,.75,.95,1.01,np.inf],
          anchor=[-np.inf,0,.25,.5,.75,.95,1.01,np.inf],
          mono_gap=[0,.1,.25,.5,np.inf],similarity=[0,.3,.5,.7,.9,1.000001],
          drug_support=[0,1,10,100,1000,np.inf],cell_support=[0,1,10,100,1000,np.inf])


def nearest_similarity(fingerprints, train_indices, batch=128):
    if not len(train_indices):raise ValueError('Empty train drug inventory')
    x=np.asarray(fingerprints,dtype=np.float32); train=x[train_indices]
    out=[]
    for start in range(0,len(x),batch):
        query=x[start:start+batch];inter=query@train.T
        union=query.sum(1)[:,None]+train.sum(1)[None,:]-inter
        scores=np.divide(inter,union,out=np.ones_like(inter),where=union>0)
        out.extend(scores.max(1).tolist())
    return np.asarray(out)


def surface_metrics(frame, predictions):
    rows=[]
    for model in predictions:
        residual=frame.target.to_numpy(dtype=float)-frame[model].to_numpy(dtype=float)
        block=pd.DataFrame(dict(surface=frame.surface_id.to_numpy(),residual=residual,squared=residual**2))
        stats=block.groupby('surface').agg(mse=('squared','mean'),bias=('residual','mean'))
        rows.append(dict(model=model,rmse=float(np.sqrt(stats.mse).mean()),bias=float(stats.bias.mean()),
                         mae_surface_bias=float(stats.bias.abs().mean())))
    return rows


def diagnose(output):
    output=Path(output).resolve()
    if not output.is_relative_to(f.ROOT/'runs/audit_revision'):raise ValueError('Invalid output scope')
    output.mkdir(parents=True,exist_ok=False)
    stage=f.ROOT/'runs/audit_revision/staged_screen_v1'
    plan=f.read_json(stage/'plan.json');f.check_seal(plan)
    parent_path=Path(plan['parent_plan']['path']);parent=f.read_json(parent_path);f.check_seal(parent)
    source_files=[Path(__file__).resolve(),f.ROOT/'docs/RESIDUAL_DIAGNOSTIC_PROTOCOL.md',f.ROOT/'tests/test_residual_diagnostics.py']
    source_records=[f.learned._file_record(p) for p in source_files]
    inputs=[f.learned._file_record(stage/'plan.json'),f.learned._file_record(parent_path)]
    config=f.load_benchmark_config(f.ROOT/plan['config']['benchmark_config'])
    drug_path=config.nci_processed_dir/'drugs.csv';inputs.append(f.learned._file_record(drug_path))
    drugs=pd.read_csv(drug_path,keep_default_na=False)
    rows=[];surface_rows=[];read_trace=[]
    for fold in range(5):
        print(f'Diagnosing fold {fold}',flush=True)
        base=parent_path.parent if fold==0 else stage
        owner=parent if fold==0 else plan
        frames={}
        for model in MODELS:
            job=next(j for j in owner['inventory']['jobs'] if j['track']=='main' and j['fold']==fold and j['seed']==41 and j['model']==model)
            path=base/'jobs'/job['id']/'receipt.json';receipt=f.read_json(path)
            f.verify_receipts({'jobs':[job]},[receipt],owner['manifest_sha256'])
            inputs += [f.learned._file_record(path),receipt['prediction']]
            frames[model]=pd.read_parquet(receipt['prediction']['path']);read_trace.append(receipt['prediction']['path'])
        f.validate_paired_points(list(frames.values()))
        data=frames[MODELS[0]].sort_values('source_id').rename(columns={'prediction':MODELS[0]}).drop(columns='model')
        for model in MODELS[1:]:data[model]=frames[model].sort_values('source_id').prediction.to_numpy()
        physical={}
        for role in ('train','validation'):
            record=plan['inputs'][str(fold)][role];f.verify([record]);inputs.append(record)
            if Path(record['path']).name!=role+'.parquet':raise ValueError('Unexpected physical response file')
            physical[role]=pd.read_parquet(record['path']);read_trace.append(record['path'])
        train=physical['train'];val=physical['validation']
        if val.source_combodrugseq.duplicated().any():raise ValueError('Duplicate source metadata')
        fields=['source_combodrugseq','surface_replicate_id','prediction_group_id','response','drug_a_id','drug_b_id','cell_id','mono_response_a','mono_response_b']
        data=data.merge(val[fields],left_on='source_id',right_on='source_combodrugseq',how='left',validate='one_to_one',indicator=True)
        if not data['_merge'].eq('both').all():raise ValueError('Prediction source not in validation')
        np.testing.assert_allclose(data.target,data.response,rtol=1e-6,atol=1e-7)
        if not data.surface_id.eq(data.surface_replicate_id).all() or not data.group_id.eq(data.prediction_group_id).all():raise ValueError('Group/surface mismatch')
        data['hsa']=np.minimum(data.mono_response_a,data.mono_response_b)
        # Frozen worker uses float32 mono covariates; retain this numerical convention.
        data['hsa']=data.hsa.astype('float32').astype(float)
        data['anchor']=data.hsa;data['mono_gap']=abs(data.mono_response_a-data.mono_response_b)
        train=train[train.primary_metric_mask & train.response_mask]
        entity=pd.concat([train[['drug_a_id','prediction_group_id']].rename(columns={'drug_a_id':'drug'}),
                          train[['drug_b_id','prediction_group_id']].rename(columns={'drug_b_id':'drug'})])
        support=entity.groupby('drug').prediction_group_id.nunique()
        data['drug_support']=np.minimum(data.drug_a_id.map(support).fillna(0),data.drug_b_id.map(support).fillna(0))
        data['cell_support']=data.cell_id.map(train.groupby('cell_id').prediction_group_id.nunique()).fillna(0)
        required=sorted(set(entity.drug)|set(data.drug_a_id)|set(data.drug_b_id))
        ids,fp,_=f.learned.build_morgan_bank(drugs,required)
        seen=set(entity.drug);indices=[i for i,d in enumerate(ids) if d in seen]
        nearest=dict(zip(ids,nearest_similarity(fp.numpy(),indices)))
        data['similarity']=np.minimum(data.drug_a_id.map(nearest),data.drug_b_id.map(nearest))
        for factor,edges in [('all',None),*BINS.items()]:
            labels=pd.Series('all',index=data.index) if edges is None else pd.cut(data[factor],edges,right=False).astype(str)
            for label,part in data.groupby(labels,sort=True):
                for metrics in surface_metrics(part,[*MODELS,'hsa']):
                    rows.append(dict(fold=fold,factor=factor,bin=str(label),points=len(part),surfaces=part.surface_id.nunique(),
                                     clusters=part.cluster_id.nunique(),**metrics))
        # Hindsight error magnitude, not a target for a model fitted on this validation set.
        data['residual']=data.target-data[MODELS[0]];data['squared']=data.residual**2
        surfaces=data.groupby('surface_id').agg(group_id=('group_id','first'),cluster_id=('cluster_id','first'),
            cell_id=('cell_id','first'),mse=('squared','mean'),bias=('residual','mean'),response=('target','mean'),
            anchor=('anchor','mean'),similarity=('similarity','min'),drug_support=('drug_support','min'))
        surfaces['rmse']=np.sqrt(surfaces.mse);surfaces['fold']=fold;surface_rows.append(surfaces.reset_index())
    result=pd.DataFrame(rows);surfaces=pd.concat(surface_rows,ignore_index=True)
    result.to_parquet(output/'strata.parquet',index=False);surfaces.to_parquet(output/'surface_residuals.parquet',index=False)
    candidates=[]
    for (factor,label),part in result[result.factor!='all'].groupby(['factor','bin']):
        if part.fold.nunique()!=5 or part.surfaces.min()<100 or part.clusters.min()<5:continue
        pivot=part.pivot(index='fold',columns='model',values='rmse')
        baseline=pivot[MODELS[0]]
        for model in MODELS[1:]:
            gain=float(1-pivot[model].mean()/baseline.mean());wins=int((pivot[model]<baseline).sum())
            candidates.append(dict(factor=factor,bin=label,model=model,winning_folds=wins,relative_gain=gain,
                passes_exploratory_screen=wins>=4 and gain>=.01,label_based=factor=='response'))
    f.verify(source_records+inputs)
    manifest=f.seal(output/'diagnostic.json',dict(scope='posthoc_development_no_fit_no_causal_claim',sources=source_records,
        inputs=inputs,read_trace=read_trace,test_response_files_read=0,new_models_trained=0,
        strata=f.learned._file_record(output/'strata.parquet'),surface_residuals=f.learned._file_record(output/'surface_residuals.parquet'),
        exploratory_subsets=candidates))
    print(f'DONE: {len(result)} stratified rows; {len(surfaces)} surfaces; '+
          f'{sum(x["passes_exploratory_screen"] for x in candidates)} exploratory subset/model signals',flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    diagnose(p.parse_args().output_dir)
