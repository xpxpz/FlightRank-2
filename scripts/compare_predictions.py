"""Compare two scored validation files on identical sessions and labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl

p=argparse.ArgumentParser()
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--bootstrap',type=int,default=2000)
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
if a.bootstrap < 0: p.error('bootstrap must be nonnegative')
def read(path,name):
    d=pl.read_parquet(path).select('Id','ranker_id','selected','score')
    if d['Id'].n_unique()!=d.height or d['ranker_id'].null_count() or d['score'].null_count() or not d['score'].is_finite().all():
        raise ValueError(f'{name}: duplicate Id, null group or invalid score')
    if d.filter(~pl.col('selected').is_in([0,1]) | pl.col('selected').is_null()).height:
        raise ValueError(f'{name}: invalid labels')
    if d.group_by('ranker_id').agg(pl.col('selected').sum().alias('n')).filter(pl.col('n') != 1).height:
        raise ValueError(f'{name}: each group must have exactly one positive')
    return d
base=read(a.baseline,'baseline'); cand=read(a.candidate,'candidate')
joined=base.join(cand,on='Id',how='full',coalesce=True,suffix='_candidate',validate='1:1')
if joined.height != base.height or joined.height != cand.height or joined.filter(pl.col('ranker_id').is_null() | pl.col('ranker_id_candidate').is_null() | (pl.col('ranker_id') != pl.col('ranker_id_candidate')) | (pl.col('selected') != pl.col('selected_candidate'))).height:
    raise ValueError('validation rows, groups or labels do not match')
def hits(d,score):
    return (d.sort(['ranker_id',score,'Id'],descending=[False,True,False])
            .group_by('ranker_id',maintain_order=True)
            .agg(pl.len().alias('size'),pl.col('selected').head(3).max().alias('hit'))
            .filter(pl.col('size') > 10).sort('ranker_id'))
x=hits(joined,'score'); y=hits(joined,'score_candidate')
if x.height == 0 or not x['ranker_id'].equals(y['ranker_id']):
    raise ValueError('no matching eligible sessions')
b=x['hit'].to_numpy().astype(float); c=y['hit'].to_numpy().astype(float); delta=c-b
result={'eligible_sessions':len(b),'baseline_hitrate_at_3':float(b.mean()),'candidate_hitrate_at_3':float(c.mean()),'delta_candidate_minus_baseline':float(delta.mean()),'bootstrap_samples':a.bootstrap,'seed':a.seed}
if a.bootstrap:
    rng=np.random.default_rng(a.seed)
    boot=np.array([rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(a.bootstrap)])
    result['paired_bootstrap_95_percent_interval']=[float(v) for v in np.quantile(boot,[.025,.975])]
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
