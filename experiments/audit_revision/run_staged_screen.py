"""Post-observation development screen: unchanged workers, 56 explicit jobs."""
import argparse
from pathlib import Path
import shutil

import pandas as pd
import run_formal as f
import run_progress as observer

PARENT = f.ROOT/'runs/audit_revision/prelaunch_v1/prepared_formal'
PROTOCOL = f.ROOT/'docs/STAGED_SCREEN_PROTOCOL.md'


def select_jobs(plan):
    jobs = [j for j in plan['inventory']['jobs']
            if j['track']=='main' and j['fold'] in (1,2,3,4) and j['seed']==41]
    from bisurfgnn.launch_plan import MAIN
    expected = {(fold, model) for fold in (1,2,3,4) for model in MAIN}
    if len(jobs)!=56 or {(j['fold'],j['model']) for j in jobs}!=expected:
        raise ValueError('Incomplete or duplicate 4-fold x 14-model screen')
    return jobs


def prepare(output, test_log):
    output=Path(output).resolve()
    if not output.is_relative_to(f.ROOT/'runs/audit_revision'):
        raise ValueError('Output must be under audit_revision')
    parent=f.read_json(PARENT/'plan.json'); f.check_seal(parent)
    ready=f.read_json(PARENT.parent/'readiness.json'); f.check_seal(ready)
    v3=f.read_json(PARENT/'progress_overlay_v3.json'); f.check_seal(v3)
    f.verify(parent['sources']+parent['snapshots']+ready['evidence']+v3['sources']+v3['acceptance'])
    jobs=select_jobs(parent)
    if shutil.disk_usage(output.parent).free < 26_000_000_000:
        raise RuntimeError('Screen storage estimate plus reserve unavailable')
    references=[]
    for j in parent['inventory']['jobs']:
        if j['track']=='main' and j['fold']==0 and j['seed']==41:
            path=PARENT/'jobs'/j['id']/'receipt.json'; receipt=f.read_json(path)
            f.verify_receipts({'jobs':[j]},[receipt],parent['manifest_sha256'])
            references.append(dict(job=j,receipt=f.learned._file_record(path)))
    if len(references)!=14: raise ValueError('Missing fold0 reference')
    output.mkdir(parents=True,exist_ok=False)
    payload={k:v for k,v in parent.items() if k!='manifest_sha256'}
    payload.update(version='post_observation_staged_screen_v1',inventory=dict(jobs=jobs,aliases=[]),
        parent_plan=f.learned._file_record(PARENT/'plan.json'),fold0_references=references,
        claim_scope='post_observation_development_screen_single_seed_no_confirmation')
    plan=f.seal(output/'plan.json',payload)
    # Fresh dispatcher base, not a forged copy of historical execution records.
    f.learned._write_json_exclusive(output/'dispatch_plan.json',dict(version='fresh_staged_dispatch_base_v1',
        formal_plan_hash=plan['manifest_sha256'],job_ids=[j['id'] for j in jobs]))
    additions=[Path(__file__).resolve(),PROTOCOL,f.ROOT/'tests/test_staged_screen.py']
    evidence=[f.learned._file_record(Path(test_log).resolve())]
    for path in additions:
        dest=output/'source_snapshot'/path.relative_to(f.ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dest)
        evidence += [f.learned._file_record(path),f.learned._file_record(dest)]
    f.seal(output/'observer.json',dict(version='staged_screen_observer_v1',
        formal_plan_hash=plan['manifest_sha256'],sources=v3['sources'],
        acceptance=[f.learned._file_record(PARENT/'progress_overlay_v3.json'),*v3['acceptance'],*evidence],
        inherited_overlays=[],note='Identical v3 observer; separate post-observation task subset.'))
    print(f'Prepared {len(jobs)} jobs. No training started.',flush=True)


def summarize(plan_path):
    plan_path=Path(plan_path).resolve(); root=plan_path.parent
    plan=f.read_json(plan_path);f.check_seal(plan)
    expected={root/'jobs'/j['id']/'receipt.json' for j in plan['inventory']['jobs']}
    if set((root/'jobs').glob('*/receipt.json'))!=expected:
        raise ValueError('Missing or extra stage receipt')
    parent=f.read_json(plan['parent_plan']['path']);f.check_seal(parent);f.verify([plan['parent_plan']])
    records=[]
    for ref in plan['fold0_references']:
        f.verify([ref['receipt']]);records.append((ref['job'],f.read_json(ref['receipt']['path']),parent['manifest_sha256']))
    records += [(j,f.read_json(root/'jobs'/j['id']/'receipt.json'),plan['manifest_sha256']) for j in plan['inventory']['jobs']]
    frames=[];seen={};receipt_evidence=[]
    for job,receipt,digest in records:
        f.verify_receipts({'jobs':[job]},[receipt],digest)
        point=pd.read_parquet(receipt['prediction']['path'])
        if set(point.fold)!={job['fold']} or set(point.seed)!={41} or set(point.model)!={job['model']}:
            raise ValueError('Point job metadata mismatch')
        if job['fold'] in seen:f.validate_paired_points([seen[job['fold']],point])
        else:seen[job['fold']]=point
        errors=pd.read_parquet(receipt['errors']['path'])
        if set(errors.model)!={job['model'],'hsa'}:raise ValueError('Error ledger models mismatch')
        frames.append(errors)
        receipt_evidence.append(dict(job=job,receipt_hash=receipt['manifest_sha256']))
    ledger=pd.concat(frames,ignore_index=True)
    anchor=ledger[ledger.model=='hsa']
    if anchor.groupby(['fold','seed','surface_id']).rmse.nunique().max()!=1:
        raise ValueError('Inconsistent HSA')
    ledger=pd.concat([ledger[ledger.model!='hsa'],anchor.drop_duplicates(['fold','seed','surface_id'])])
    means=ledger.groupby(['model','fold']).rmse.mean().unstack()
    if len(means)!=15 or means.shape[1]!=5 or means.isna().any().any():raise ValueError('Incomplete main table')
    result={str(model):dict(by_fold={str(k):float(v) for k,v in row.items()},mean_rmse=float(row.mean())) for model,row in means.iterrows()}
    f.seal(root/'screen_summary.json',dict(scope='descriptive_single_seed_development_not_confirmatory',
        plan_hash=plan['manifest_sha256'],models=result,receipts=receipt_evidence,
        next_step='Discuss branch using STAGED_SCREEN_PROTOCOL; no automatic additional experiments.'))
    print('Screen complete; descriptive summary saved; no further tasks launched.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','summarize'])
    p.add_argument('--output-dir',type=Path);p.add_argument('--test-log',type=Path);p.add_argument('--plan',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output_dir,a.test_log)
    elif a.action=='summarize':summarize(a.plan)
    else:
        observer.launch(a.plan,PARENT.parent/'readiness.json',a.plan.parent/'observer.json')
        summarize(a.plan)
