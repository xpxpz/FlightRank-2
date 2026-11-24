"""Run baseline v5 feature code with a session-safe common GroupKFold split.

The feature cells are executed verbatim from the saved public notebook. Model
training and splitting are adapted so its validation session IDs can match NB0036.
Optionally replace leaky carrier means with nested cross-fitted means.
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
p.add_argument('--fold', type=int, default=0, choices=range(10))
p.add_argument('--split', choices=['groupkfold', 'original'], default='groupkfold')
p.add_argument('--split-file', type=Path, help='parquet ranker_id/fold; validation fold is 0')
p.add_argument('--carrier-stats', choices=['original', 'nested'], default='original')
p.add_argument('--rounds', type=int, default=1000)
p.add_argument('--threads', type=int, default=1)
a = p.parse_args()
os.environ.setdefault('POLARS_MAX_THREADS', str(a.threads))
os.environ.setdefault('OMP_NUM_THREADS', str(a.threads))
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from metrics import hitrate_at_3
from provenance import capture

a.out.mkdir(parents=True, exist_ok=True)
start = time.monotonic()
provenance = capture(a.data)
n = json.loads((Path(__file__).resolve().parents[1] / 'references/baseline.ipynb').read_text())
ns = {'pl': pl, 'np': np}
# These columns are excluded by baseline cell 11 and are never consumed by
# feature cell 8. Keep duration and marketing carrier for segment counts.
prune = ['profileId', 'miniRules0_percentage', 'miniRules1_percentage', 'pricingInfo_passengerCount']
for leg in (0, 1):
    for seg in (2, 3):
        for suffix in ('aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                       'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                       'cabinClass', 'departureFrom_airport_iata', 'flightNumber',
                       'operatingCarrier_code', 'seatsAvailable'):
            prune.append(f'legs{leg}_segments{seg}_{suffix}')

print('stage: read train', flush=True)
train_schema = pl.scan_parquet(a.data / 'train.parquet').collect_schema()
train = pl.read_parquet(a.data / 'train.parquet', columns=[c for c in train_schema if c not in prune])
print('stage: read test', flush=True)
test_schema = pl.scan_parquet(a.data / 'test.parquet').collect_schema()
test = pl.read_parquet(a.data / 'test.parquet', columns=[c for c in test_schema if c not in prune])
if train['ranker_id'].null_count() or test['ranker_id'].null_count():
    raise ValueError('ranker_id must not be null')
if train.filter(~pl.col('selected').is_in([0, 1]) | pl.col('selected').is_null()).height:
    raise ValueError('selected must be binary and non-null')
bad = train.group_by('ranker_id').agg(pl.col('selected').sum().alias('n')).filter(pl.col('n') != 1)
if bad.height:
    raise ValueError(f'{bad.height} train sessions do not have exactly one positive')
print('stage: assign folds', flush=True)
# Reproduce GroupKFold on per-session counts without creating 18M Python strings.
# sklearn 1.9 uses np.unique's sorted group order and stable descending size.
group_counts = train.group_by('ranker_id').len().sort('ranker_id')
if a.split_file:
    fold_lookup = pl.read_parquet(a.split_file).select('ranker_id','fold')
    if fold_lookup['ranker_id'].n_unique() != fold_lookup.height or fold_lookup.filter(~pl.col('fold').is_in([0, 1]) | pl.col('fold').is_null()).height:
        raise ValueError('split file must assign each ranker_id a unique fold 0 or 1')
    assigned = group_counts.join(fold_lookup, on='ranker_id', how='left')
    if assigned['fold'].null_count() or set(assigned['fold'].unique().to_list()) != {0, 1}:
        raise ValueError('split file must assign all train sessions and contain both folds')
elif a.split == 'original':
    boundary = 16487352
    if len(train) <= boundary:
        raise ValueError('original fixed-row split requires the complete training data')
    assigned = (train.select('ranker_id').with_row_index('_row')
                .group_by('ranker_id').agg(pl.col('_row').min().alias('lo'),pl.col('_row').max().alias('hi')))
    if assigned.filter((pl.col('lo') < boundary) & (pl.col('hi') >= boundary)).height:
        raise ValueError('split would cut a search session')
    fold_lookup = assigned.select('ranker_id', pl.when(pl.col('lo') < boundary).then(1).otherwise(0).cast(pl.Int8).alias('fold'))
else:
    weights = group_counts['len'].to_numpy()
    order = np.argsort(weights, kind='stable')[::-1]
    totals = np.zeros(10)
    group_folds = np.empty(len(weights), dtype=np.int8)
    for idx in order:
        f = np.argmin(totals)
        totals[f] += weights[idx]
        group_folds[idx] = f
    fold_lookup = group_counts.select('ranker_id').with_columns(pl.Series('fold', group_folds))
validation_fold = 0 if a.split_file or a.split == 'original' else a.fold
print('stage: project inputs', flush=True)
train = train.drop('__index_level_0__', strict=False)
test = test.drop('__index_level_0__', strict=False).with_columns(pl.lit(0, dtype=pl.Int64).alias('selected'))
ntrain = len(train)
# Carrier means are the only uses of train inside the original feature cell.
aggregate_train = train.select('ranker_id', 'selected', 'legs0_segments0_marketingCarrier_code', 'legs1_segments0_marketingCarrier_code')
print('stage: concatenate', flush=True)
raw = pl.concat((train.drop(prune, strict=False), test.drop(prune, strict=False))).with_row_index('_input_row')
ns.update(train=aggregate_train, data_raw=raw)
if a.carrier_stats == 'nested':
    # Training rows receive inner out-of-fold carrier means. Validation and
    # test rows receive outer-training-only means. No validation labels enter.
    outer_train = aggregate_train.join(fold_lookup, on='ranker_id', how='left').filter(pl.col('fold') != validation_fold).drop('fold')
    # Match unshuffled GroupKFold from per-session counts, avoiding a Python
    # string for each of the ~16M outer-training rows.
    inner_counts = outer_train.group_by('ranker_id').len().sort('ranker_id')
    weights = inner_counts['len'].to_numpy()
    order = np.argsort(weights, kind='stable')[::-1]
    totals = np.zeros(5)
    group_folds = np.empty(len(weights), dtype=np.int8)
    for idx in order:
        f = np.argmin(totals)
        totals[f] += weights[idx]
        group_folds[idx] = f
    inner_lookup = inner_counts.select('ranker_id').with_columns(pl.Series('_inner_fold', group_folds))
    inner_train = outer_train.join(inner_lookup, on='ranker_id', how='left')
    outer_lookup = fold_lookup.rename({'fold':'_outer_fold'})
    def apply_carrier_means(target, source):
        for leg in (0, 1):
            col = f'legs{leg}_segments0_marketingCarrier_code'
            alias = f'carrier{leg}_pop'
            stats = source.group_by(col).agg(pl.col('selected').mean().alias(alias))
            target = target.join(stats, on=col, how='left').with_columns(pl.col(alias).fill_null(0.0))
        return target.select('_input_row', 'carrier0_pop', 'carrier1_pop')
    def safe_carrier_features(df):
        # Split only row IDs and grouping keys, not the full wide feature table.
        cols = ['_input_row','ranker_id','legs0_segments0_marketingCarrier_code','legs1_segments0_marketingCarrier_code']
        narrow = df.select(cols)
        tr_part = narrow.filter(pl.col('_input_row') < ntrain).join(outer_lookup, on='ranker_id', how='left')
        train_rows = tr_part.filter(pl.col('_outer_fold') != validation_fold).join(inner_lookup, on='ranker_id', how='left')
        pieces = []
        for j in range(5):
            target = train_rows.filter(pl.col('_inner_fold') == j)
            source = inner_train.filter(pl.col('_inner_fold') != j)
            pieces.append(apply_carrier_means(target, source))
        validation_rows = tr_part.filter(pl.col('_outer_fold') == validation_fold)
        test_rows = narrow.filter(pl.col('_input_row') >= ntrain)
        pieces.append(apply_carrier_means(validation_rows, outer_train))
        pieces.append(apply_carrier_means(test_rows, outer_train))
        features = pl.concat(pieces)
        if features.height != df.height or features['_input_row'].n_unique() != df.height:
            raise ValueError('carrier features must cover each row exactly once')
        return df.join(features, on='_input_row', how='left', maintain_order='left')
    ns['safe_carrier_features'] = safe_carrier_features
del train, test, raw, aggregate_train
gc.collect()
for i in (8, 9, 11):
    print(f'stage: baseline feature cell {i}', flush=True)
    source = ''.join(n['cells'][i]['source'])
    if i == 8 and a.carrier_stats == 'nested':
        start_marker = '# Popularity features - efficient join'
        end_marker = '# Final features including popularity'
        start_at, end_at = source.index(start_marker), source.index(end_marker)
        source = source[:start_at] + 'df = safe_carrier_features(df)\n\n' + source[end_at:]
    exec(compile(source, f'baseline-cell-{i}', 'exec'), ns)
# Feature joins need not preserve row order; restore it before splitting.
data = ns['data'].sort('_input_row')
features = [c for c in ns['feature_cols'] if c != '_input_row']
cat_features = ns['cat_features_final']
# Release intermediate feature tables from the notebook namespace.
ns.clear()
if a.carrier_stats == 'nested':
    del outer_train, inner_train, inner_counts, inner_lookup, outer_lookup
gc.collect()
print('stage: encode categories', flush=True)
data = data.with_columns([(pl.col(c).rank('dense') - 1).fill_null(-1).cast(pl.Int32) for c in cat_features])
# Preserve original data row order, then group contiguous explicitly.
train_data = data[:ntrain].join(fold_lookup, on='ranker_id', how='left').sort(['ranker_id', 'Id'])
test_data = data[ntrain:].with_row_index('_original_row').sort(['ranker_id', 'Id'])
tr = train_data.filter(pl.col('fold') != validation_fold)
va = train_data.filter(pl.col('fold') == validation_fold)
del data, train_data
gc.collect()
def dmatrix(frame, label=True):
    sizes = frame.group_by('ranker_id', maintain_order=True).len()['len'].to_numpy()
    # DMatrix stores feature values as float32; cast before materializing to
    # avoid a full-size float64 NumPy matrix and redundant copy.
    x = frame.select([pl.col(c).cast(pl.Float32, strict=False) for c in features]).to_numpy()
    if x.dtype.kind not in 'fiub':
        raise ValueError(f'non-numeric feature matrix: {x.dtype}')
    x = np.nan_to_num(x, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    return xgb.DMatrix(x, label=frame['selected'].to_numpy() if label else None,
                       group=sizes, feature_names=features, nthread=a.threads)
print('stage: construct matrices', flush=True)
train_rows = len(tr)
dtr = dmatrix(tr)
del tr
gc.collect()
dva = dmatrix(va)
params = {'objective':'rank:pairwise', 'eval_metric':'ndcg@3', 'max_depth':10,
          'min_child_weight':10, 'subsample':0.8, 'colsample_bytree':0.8,
          'lambda':10.0, 'learning_rate':0.05, 'seed':42, 'nthread':a.threads}
model = xgb.train(params, dtr, num_boost_round=a.rounds, evals=[(dtr,'train'),(dva,'val')], verbose_eval=50)
score = model.predict(dva)
metric = hitrate_at_3(va['ranker_id'].to_numpy(), va['selected'].to_numpy(), score)
va.select('Id','ranker_id','selected','fold').with_columns(pl.Series('score',score)).write_parquet(a.out / 'validation.parquet')
model.save_model(a.out / 'model.json')
pred = model.predict(dmatrix(test_data, label=False))
test_data.select('_original_row','Id','ranker_id').with_columns(pl.Series('score',pred)).sort('_original_row').drop('_original_row').write_parquet(a.out / 'test_scores.parquet')
summary = {'model':'baseline_v5', 'provenance':provenance, 'source_version':250405784, 'split':('explicit_file' if a.split_file else a.split), 'split_file':str(a.split_file) if a.split_file else None, 'carrier_stats':a.carrier_stats, 'fold':validation_fold,
           'rounds':a.rounds, 'params':params, 'feature_names':features,
           'train_rows':train_rows, 'validation_rows':len(va), 'hitrate_at_3':metric,
           'seconds':time.monotonic()-start, 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024), 'xgboost':xgb.__version__, 'polars':pl.__version__,
           'changes':['session-safe split: original boundary or GroupKFold for comparison', 'restore input order after feature joins', 'early projection of unused columns', 'release unused intermediate tables', 'construct float32 matrix before DMatrix', 'explicit group sort', 'thread limit', 'save model and scores'] + (['nested cross-fitted carrier means with narrow table and compact folds'] if a.carrier_stats == 'nested' else [])}
(a.out / 'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({k:v for k,v in summary.items() if k != 'feature_names'}, indent=2))
