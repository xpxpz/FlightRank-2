"""Combine NB0036 fold predictions and build a validated ranked submission."""
import argparse
import json
from pathlib import Path
import numpy as np
import polars as pl
from metrics import hitrate_at_3

p=argparse.ArgumentParser()
p.add_argument('--fold-root',type=Path,required=True)
p.add_argument('--test',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--allow-mixed-device',action='store_true',help='allow CPU/PPU folds while preserving per-fold provenance')
p.add_argument('--folds',default='all',help='all, or comma-separated fold ids; subsets explicitly marked partial')
a=p.parse_args()
folds=list(range(10)) if a.folds=='all' else [int(x) for x in a.folds.split(',')]
if not folds or len(set(folds))!=len(folds) or any(i<0 or i>9 for i in folds):
    p.error('choose unique folds in 0..9')
a.output.mkdir(parents=True,exist_ok=True)
from provenance import digest
source=pl.read_parquet(a.test,columns=['Id','ranker_id'])
if source['Id'].n_unique()!=source.height: raise ValueError('test Id is not unique')
merged=source.with_row_index('_row'); oof=[]; model_params=None
test_sha256 = digest(a.test)
fold_signatures = {}
def comparable(sig):
    value = json.loads(json.dumps(sig))
    if a.allow_mixed_device:
        value['params'].pop('device', None)
        for k in ('xgboost_build','libxgboost_sha256','implementation_note'):
            value['provenance'].pop(k, None)
    return value
for i in folds:
    folder=a.fold_root/f'fold_{i}'
    info=json.loads((folder/'result.json').read_text())
    signature=info['signature']
    if signature['provenance']['inputs']['test']['sha256'] != test_sha256:
        raise ValueError(f'fold {i} test data checksum does not match')
    fold_signatures[str(i)] = signature
    if model_params is None: model_params=signature
    elif comparable(signature) != comparable(model_params): raise ValueError(f'fold {i} configuration differs from other folds')
    pred=pl.read_parquet(folder/'test_scores.parquet').select('Id','ranker_id','score')
    if pred.height!=source.height or pred['Id'].n_unique()!=pred.height or pred['score'].null_count() or not pred['score'].is_finite().all():
        raise ValueError(f'fold {i} invalid test predictions')
    merged=merged.join(pred.rename({'ranker_id':f'_group_{i}','score':f'_score_{i}'}),on='Id',how='left',validate='1:1')
    if merged[f'_group_{i}'].null_count() or merged.filter(pl.col('ranker_id')!=pl.col(f'_group_{i}')).height:
        raise ValueError(f'fold {i} test Id/ranker mismatch')
    val=pl.read_parquet(folder/'validation.parquet')
    if val['fold'].unique().to_list()!=[i]: raise ValueError(f'fold {i} validation fold mismatch')
    oof.append(val)
if merged.height != source.height: raise ValueError('test row count changed')
merged=merged.with_columns(pl.mean_horizontal([f'_score_{i}' for i in folds]).alias('score')).sort('_row')
ranked=merged.with_columns(pl.col('score').rank('ordinal',descending=True).over('ranker_id').cast(pl.Int64).alias('selected'))
check=ranked.group_by('ranker_id').agg(pl.len().alias('n'),pl.col('selected').n_unique().alias('uniq'),pl.col('selected').min().alias('lo'),pl.col('selected').max().alias('hi'))
if check.filter((pl.col('n')!=pl.col('uniq')) | (pl.col('lo')!=1) | (pl.col('hi')!=pl.col('n'))).height:
    raise ValueError('invalid ranks')
ranked.select('Id','ranker_id','selected').write_csv(a.output/'submission.csv')
ranked.select('Id','ranker_id','score','selected').write_parquet(a.output/'test_predictions.parquet')
oof=pl.concat(oof).sort('Id')
if oof['Id'].n_unique()!=oof.height or oof.group_by('ranker_id').agg(pl.col('fold').n_unique().alias('n')).filter(pl.col('n')!=1).height:
    raise ValueError('duplicate OOF Id or session across folds')
oof.write_parquet(a.output/'oof.parquet')
metric=hitrate_at_3(oof['ranker_id'].to_numpy(),oof['selected'].to_numpy(),oof['score'].to_numpy())
summary={'folds':folds,'complete_10_fold':set(folds)==set(range(10)),'signature':model_params,'fold_signatures':fold_signatures,'mixed_device_allowed':a.allow_mixed_device,'oof_rows':oof.height,'oof_hitrate_at_3':metric,'test_rows':source.height}
(a.output/'result.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
