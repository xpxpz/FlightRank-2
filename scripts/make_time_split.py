"""Create an outer session split by request time, without timestamp ties crossing."""
import argparse
import json
from pathlib import Path
import polars as pl

p=argparse.ArgumentParser()
p.add_argument('--train', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--validation-fraction', type=float, default=0.1)
a=p.parse_args()
if not 0 < a.validation_fraction < 1:
    p.error('validation fraction must be between 0 and 1')
q=pl.scan_parquet(a.train).select('ranker_id','requestDate')
schema=q.collect_schema()
if schema['requestDate'] == pl.String:
    q=q.with_columns(pl.col('requestDate').str.to_datetime(strict=False))
elif not isinstance(schema['requestDate'], (pl.Datetime, pl.Date)):
    raise ValueError(f'unsupported requestDate dtype: {schema["requestDate"]}')
g=q.group_by('ranker_id').agg(pl.col('requestDate').min().alias('time'),
                               pl.col('requestDate').n_unique().alias('n_times'),
                               pl.col('requestDate').null_count().alias('nulls')).collect()
if g.filter(pl.col('ranker_id').is_null() | (pl.col('nulls') > 0) | (pl.col('n_times') != 1)).height:
    raise ValueError('every session needs one consistent non-null request time')
g=g.sort(['time','ranker_id'])
idx=min(g.height-1,max(1,int(g.height*(1-a.validation_fraction))))
cutoff=g['time'][idx]
g=g.with_columns(pl.when(pl.col('time') >= cutoff).then(pl.lit(0)).otherwise(pl.lit(1)).cast(pl.Int8).alias('fold'))
if g.filter(pl.col('fold')==1).height < 10 or g.filter(pl.col('fold')==0).height == 0:
    raise ValueError('split does not leave at least ten training sessions and one validation session')
a.output.parent.mkdir(parents=True,exist_ok=True)
g.select('ranker_id','fold').write_parquet(a.output)
print(json.dumps({'cutoff':str(cutoff),'train_sessions':g.filter(pl.col('fold')==1).height,'validation_sessions':g.filter(pl.col('fold')==0).height,'time_ties_kept_together':True},indent=2))
