"""Extract reproducible complete search sessions; preserve source row order."""
import argparse
import json
from pathlib import Path
import polars as pl

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--sessions', type=int, default=1000)
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
if a.sessions < 10:
    p.error('--sessions must be at least 10')
a.output.mkdir(parents=True, exist_ok=True)
manifest = {'seed': a.seed, 'requested_sessions_per_set': a.sessions, 'datasets': {}}
for name in ('train', 'test'):
    source = pl.scan_parquet(a.input / f'{name}.parquet')
    ids = source.select('ranker_id').unique().collect().sort('ranker_id')
    if ids['ranker_id'].null_count():
        raise ValueError(f'{name}: null ranker_id')
    ids = ids.sample(n=min(a.sessions, ids.height), seed=a.seed, shuffle=True)
    subset = (source.with_row_index('_sample_source_row')
              .join(ids.lazy(), on='ranker_id', how='semi')
              .sort('_sample_source_row').drop('_sample_source_row').collect())
    if subset['Id'].n_unique() != subset.height:
        raise ValueError(f'{name}: Id is not unique')
    if name == 'train':
        if subset.filter(~pl.col('selected').is_in([0, 1]) | pl.col('selected').is_null()).height:
            raise ValueError('train: selected must be binary and non-null')
        bad = subset.group_by('ranker_id').agg(pl.col('selected').sum().alias('positives')).filter(pl.col('positives') != 1)
        if bad.height:
            raise ValueError(f'{bad.height} sessions have labels inconsistent with competition description')
    subset.write_parquet(a.output / f'{name}.parquet')
    manifest['datasets'][name] = {'sessions':ids.height, 'rows':subset.height}
    print(f'{name}: {ids.height} sessions, {subset.height} rows')
(a.output / 'sample_manifest.json').write_text(json.dumps(manifest, indent=2))
