"""Record dataset integrity, session boundaries and time ordering."""
import argparse
import hashlib
import json
from pathlib import Path
import polars as pl

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
report={}
for name in ['train','test']:
    path=a.data/f'{name}.parquet'
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    q=pl.scan_parquet(path).with_row_index('_row')
    schema=q.collect_schema()
    cols=['ranker_id','Id','_row']
    if 'selected' in schema: cols.append('selected')
    if 'requestDate' in schema: cols.append('requestDate')
    q=q.select(cols)
    row=q.select(pl.len().alias('rows'),pl.col('Id').n_unique().alias('unique_ids'),
                 pl.col('ranker_id').n_unique().alias('sessions'),pl.col('ranker_id').null_count().alias('null_sessions'),
                 (pl.col('ranker_id') != pl.col('ranker_id').shift(1)).fill_null(True).sum().alias('session_runs')).collect().row(0,named=True)
    expr=[pl.len().alias('size'),pl.col('_row').min().alias('first_row')]
    if 'selected' in schema:
        expr += [pl.col('selected').sum().alias('positives'),pl.col('selected').null_count().alias('null_labels'),(~pl.col('selected').is_in([0,1])).sum().alias('bad_labels')]
    if 'requestDate' in schema:
        col=pl.col('requestDate')
        if schema['requestDate']==pl.String: col=col.str.to_datetime(strict=False)
        expr += [col.min().alias('first_time'),col.n_unique().alias('distinct_times'),col.null_count().alias('null_times')]
    groups=q.group_by('ranker_id').agg(expr).collect().sort('first_row')
    row.update(file_bytes=path.stat().st_size,sha256=digest.hexdigest(),columns=len(schema)-1,
               contiguous_sessions=(row['session_runs']==row['sessions']),
               sessions_at_most_10=groups.filter(pl.col('size') <= 10).height)
    if 'selected' in schema:
        row['invalid_label_sessions']=groups.filter((pl.col('positives') != 1) | (pl.col('null_labels')>0) | (pl.col('bad_labels')>0)).height
    if 'requestDate' in schema:
        row['invalid_time_sessions']=groups.filter((pl.col('distinct_times')!=1) | (pl.col('null_times')>0)).height
        row['request_time_decreases']=groups.select((pl.col('first_time')<pl.col('first_time').shift(1)).fill_null(False).sum()).item()
        row['request_time_min']=str(groups['first_time'].min());row['request_time_max']=str(groups['first_time'].max())
    if name=='train' and row['rows']>16487352:
        boundary=q.filter(pl.col('_row').is_between(16487351,16487352)).select('_row','ranker_id').collect().sort('_row')
        row['baseline_fixed_boundary_same_adjacent_session']=boundary['ranker_id'][0]==boundary['ranker_id'][1]
        # Even noncontiguous sessions can cross the fixed split.
        cross=q.group_by('ranker_id').agg(pl.col('_row').min().alias('lo'),pl.col('_row').max().alias('hi')).filter((pl.col('lo') < 16487352) & (pl.col('hi') >= 16487352)).select(pl.len()).collect().item()
        row['baseline_fixed_boundary_crossing_sessions']=cross
    report[name]=row
    print(json.dumps({name:row},indent=2),flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2))
