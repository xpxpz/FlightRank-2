"""Execute NB0036 feature cells and serial XGBoost folds from the saved notebook.

This preserves the author's cross-fitted aggregations (and their limitations),
while adding sorted contiguous groups, resource controls and durable artifacts.
"""
import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--folds', default='0', help='Comma-separated folds, or all')
p.add_argument('--compact-aggregates', action='store_true', help='project aggregation source to required columns')
p.add_argument('--duration-mode', choices=['original', 'vectorized'], default='original')
p.add_argument('--test-aggregates', choices=['all', 'requested'], default='all', help='requested computes only requested test fold columns')
p.add_argument('--split-file', type=Path, help='parquet ranker_id/fold for nested mode; validation is fold 0')
p.add_argument('--aggregate-mode', choices=['original', 'nested'], default='original')
p.add_argument('--ablate', choices=['none', 'price', 'convenience', 'behavior'], default='none')
p.add_argument('--rounds', type=int, default=3500)
p.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
p.add_argument('--threads', type=int, default=1)
a = p.parse_args()
folds = list(range(10)) if a.folds == 'all' else [int(s) for s in a.folds.split(',')]
if len(set(folds)) != len(folds) or not all(0 <= f < 10 for f in folds):
    p.error('folds must be unique values from 0 through 9')
if a.split_file and (a.aggregate_mode != 'nested' or folds != [0]):
    p.error('--split-file requires --aggregate-mode nested --folds 0')
if a.aggregate_mode == 'nested' and len(folds) != 1:
    p.error('nested aggregate mode requires exactly one outer fold per run')
a.split_file = a.split_file.resolve() if a.split_file else None
a.data = a.data.resolve()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('POLARS_MAX_THREADS', str(a.threads))
os.environ.setdefault('OMP_NUM_THREADS', str(a.threads))
import numpy as np
import polars as pl
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from typing import List, Dict
from metrics import hitrate_at_3
from provenance import capture, digest

start = time.monotonic()
provenance = capture(a.data)
provenance['xgboost_build'] = xgb.build_info()
provenance['libxgboost_sha256'] = digest(xgb.build_info()['libxgboost'])
provenance['implementation_note'] = 'PPU SDK 12.6 source build, quantile Entry equality compatibility patch'
nb = json.loads((Path(__file__).resolve().parents[1] / 'references/NB0036.ipynb').read_text())
ns = dict(pl=pl, np=np, pd=pd, gc=gc, os=os, List=List, Dict=Dict, GroupKFold=GroupKFold)
# Cell 28 reads relative paths; only the working directory is adapted.
os.chdir(a.data)
for i in [4, *range(6, 30), 32, 33, 34]:
    if i in (7,):  # defined but main reads directly; retain cell for fidelity anyway
        pass
    source = ''.join(nb['cells'][i]['source'])
    if i == 22 and a.aggregate_mode == 'original' and a.test_aggregates == 'requested':
        original = 'for fold_i in range(10):'
        if source.count(original) != 1:
            raise ValueError('unexpected upstream test aggregation loop')
        source = source.replace(original, f'for fold_i in {folds!r}:')
    if i == 22 and a.aggregate_mode == 'original':
        start_fold = source.index('    gkf = GroupKFold(n_splits=10)')
        end_fold = source.index('    # --- CV Loop for Training Features ---', start_fold)
        source = source[:start_fold] + """    counts = train_df.group_by('ranker_id').len().sort('ranker_id')
    weights = counts['len'].to_numpy()
    order = np.argsort(weights, kind='stable')[::-1]
    totals = np.zeros(10)
    assigned = np.empty(len(weights), dtype=np.int8)
    for idx in order:
        fold_id = np.argmin(totals)
        totals[fold_id] += weights[idx]
        assigned[idx] = fold_id
    lookup = counts.select('ranker_id').with_columns(pl.Series('fold', assigned))
    train_df = train_df.join(lookup, on='ranker_id', how='left', maintain_order='left')
    original_train_df = train_df.clone()
    del counts, lookup, weights, order, assigned
    gc.collect()

""" + source[end_fold:]
    if i == 22 and a.aggregate_mode == 'original' and a.compact_aggregates:
        marker = '    # --- CV Loop for Training Features ---'
        assert source.count(marker) == 1
        projection = """    aggregate_columns = {'fold', 'selected'}
    for config in agg_configs.values():
        aggregate_columns.update(config['group_by'])
        aggregate_columns.add(config['agg_col'])
    aggregation_base = original_train_df.select(sorted(aggregate_columns))
    print('Projected aggregation inputs:', len(aggregate_columns), 'columns', flush=True)

"""
        source = source.replace(marker, projection + marker)
        source = source.replace("df_train_fold = original_train_df.filter(pl.col('fold') != i)", "df_train_fold = aggregation_base.filter(pl.col('fold') != i)")
        source = source.replace("df_train_for_test = original_train_df.filter(pl.col('fold') != fold_i)", "df_train_for_test = aggregation_base.filter(pl.col('fold') != fold_i)")
    if i == 33:
        # Sample data may have constant fold-aggregate features. Test only has
        # their fold-suffixed names, so ignore nonexistent unsuffixed names.
        source = source.replace('test_df.drop(zero_variance_cols)', 'test_df.drop(zero_variance_cols, strict=False)')
    print(f'Running NB0036 cell {i}', flush=True)
    exec(compile(source, f'NB0036-cell-{i}', 'exec'), ns)
    if i == 22 and a.aggregate_mode == 'nested':
        from nested_aggregates import make_nested_aggregator
        ns['create_cv_aware_aggregate_features_no_leakage'] = make_nested_aggregator(folds[0], split_file=a.split_file)
    if i == 8 and a.duration_mode == 'vectorized':
        from duration_features import convert_duration_to_seconds
        ns['convert_duration_to_seconds'] = convert_duration_to_seconds
    if i == 8:
        # Polars 1.44 enforces the declared Float64 UDF output type. The
        # original duration parser returns integers for whole seconds.
        original_duration_parser = ns['convert_mixed_time_polars']
        def duration_as_float(value):
            result = original_duration_parser(value)
            return float(result) if result is not None else None
        ns['convert_mixed_time_polars'] = duration_as_float
train, test, drop = ns['train'], ns['test'], ns['drop']
if train['ranker_id'].null_count() or train['Id'].n_unique() != train.height:
    raise ValueError('train contains null sessions or duplicate Id values after feature engineering')
if train.filter(~pl.col('selected').is_in([0, 1]) | pl.col('selected').is_null()).height:
    raise ValueError('selected must be binary and non-null')
# Restore original test ordering after author feature joins.
original_test_order = pl.read_parquet(a.data / 'test.parquet', columns=['Id']).with_row_index('_original_row')
if original_test_order['Id'].n_unique() != original_test_order.height:
    raise ValueError('test Id values must be unique')
test = test.join(original_test_order, on='Id', how='left', validate='1:1')
if test['_original_row'].null_count():
    raise ValueError('test feature rows do not match original Id values')
print(f'Feature engineering and encoding seconds: {time.monotonic()-start:.1f}', flush=True)
params = {'objective':'rank:pairwise', 'eval_metric':'ndcg@3', 'max_depth':10,
          'min_child_weight':10, 'subsample':0.8, 'colsample_bytree':0.8,
          'lambda':10.0, 'learning_rate':0.05, 'seed':42, 'nthread':a.threads,
          'lambdarank_num_pair_per_sample':8, 'tree_method':'hist', 'device':a.device}
results = []
run_signature = {'model':'NB0036', 'duration_mode':a.duration_mode, 'compact_aggregates':a.compact_aggregates, 'aggregate_mode':a.aggregate_mode, 'test_aggregates':a.test_aggregates, 'split_file':str(a.split_file) if a.split_file else None, 'ablation':a.ablate, 'rounds':a.rounds, 'params':params, 'provenance':provenance}
for fold in folds:
    fold_start = time.monotonic()
    out = a.out / f'fold_{fold}'
    out.mkdir(exist_ok=True)
    tr = train.filter(pl.col('fold') != fold).sort(['ranker_id','Id'])
    va = train.filter(pl.col('fold') == fold).sort(['ranker_id','Id'])
    if tr.group_by('ranker_id').agg(pl.col('selected').sum().alias('n')).filter(pl.col('n') != 1).height:
        raise ValueError('training sessions must contain exactly one positive')
    all_features = tr.drop(drop).columns
    aggregate_keys = set().union(*(set(ns[k]) for k in (
        'aggregation_configurations', 'high_priority_aggregation_configs',
        'tier1_aggregation_configs', 'tier2_aggregation_configs', 'tier3_aggregation_configs')))
    # Groups deliberately overlap: a price-related behavioral aggregate is
    # removed in either ablation, and exact removed names are saved.
    if a.ablate == 'price':
        removed = [c for c in all_features if any(t in c.lower() for t in ('price', 'tax', 'fare'))]
    elif a.ablate == 'convenience':
        removed = [c for c in all_features if any(t in c.lower() for t in ('duration', 'segment', 'segs', 'stop', 'direct', 'wait', 'departure', 'arrival', 'travel_time'))]
    elif a.ablate == 'behavior':
        removed = [c for c in all_features if c in aggregate_keys]
    else:
        removed = []
    features = [c for c in all_features if c not in set(removed)]
    if not features:
        raise ValueError('ablation removed every feature')
    (out / 'features.json').write_text(json.dumps({'kept':features, 'removed':removed}, indent=2))
    def matrix(frame, label=True):
        sizes = frame.group_by('ranker_id', maintain_order=True).len()['len'].to_numpy()
        x = frame.select([pl.col(c).cast(pl.Float32, strict=False) for c in features]).to_numpy()
        x = np.nan_to_num(x, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
        return xgb.DMatrix(x, label=frame['selected'].to_numpy() if label else None,
                           group=sizes, feature_names=features, nthread=a.threads)
    dtr, dva = matrix(tr), matrix(va)
    model = xgb.train(params, dtr, num_boost_round=a.rounds,
                      evals=[(dtr,'train'),(dva,'val')], verbose_eval=50)
    device_used = json.loads(model.save_config())['learner']['generic_param']['device']
    print('Actual XGBoost device:',device_used,flush=True)
    if a.device.startswith('cuda') and not device_used.startswith('cuda'):
        raise RuntimeError('GPU requested but XGBoost fell back to CPU')
    scores = model.predict(dva)
    metric = hitrate_at_3(va['ranker_id'].to_numpy(), va['selected'].to_numpy(), scores)
    va.select('Id','ranker_id','selected','fold').with_columns(pl.Series('score',scores)).write_parquet(out / 'validation.parquet')
    model.save_model(out / 'model.json')
    del dtr, dva, tr, va
    gc.collect()
    # Each fold's test aggregates are already materialized by the author code.
    suffix = f'_fold{fold}'
    renames = {c:c[:-len(suffix)] for c in test.columns if c.endswith(suffix)}
    te = test.rename(renames).sort(['ranker_id','Id'])
    preds = model.predict(matrix(te, label=False))
    te.select('_original_row','Id','ranker_id').with_columns(pl.Series('score',preds)).sort('_original_row').drop('_original_row').write_parquet(out / 'test_scores.parquet')
    result = {'signature':run_signature, 'fold':fold,'hitrate_at_3':metric,'seconds':time.monotonic()-fold_start, 'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    results.append(result)
    del te, model, preds
    gc.collect()
    print(json.dumps(result), flush=True)
summary = {'model':'NB0036', 'provenance':provenance, 'duration_mode':a.duration_mode, 'compact_aggregates':a.compact_aggregates, 'aggregate_mode':a.aggregate_mode, 'test_aggregates':a.test_aggregates, 'split_file':str(a.split_file) if a.split_file else None, 'ablation':a.ablate, 'folds':folds, 'rounds':a.rounds, 'params':params,
           'xgboost':xgb.__version__, 'polars':pl.__version__, 'results':results,
           'seconds':time.monotonic()-start, 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
           'changes':['compact group fold assignment', 'float32 DMatrix construction', 'ignore absent test columns when dropping zero-variance train columns', 'duration parser returns Float64 for Polars compatibility', 'restore test input order by Id', 'explicit group sort', 'thread limit', 'save per-fold model and predictions'] + (['nested aggregate source projection and compact fold assignment'] if a.aggregate_mode == 'nested' else []) + (['only requested test fold aggregates'] if a.test_aggregates == 'requested' and a.aggregate_mode == 'original' else []),
           'limitations':(['author cross-fitted label aggregates are not nested for outer validation'] if a.aggregate_mode == 'original' else ['nested mode isolates only label-based aggregates']) + ['not time-causal']}
(a.out / 'result.json').write_text(json.dumps(summary, indent=2))
