"""Check a vectorized duration parser against the saved author's parser."""
import argparse,json,re,sys
from pathlib import Path
import polars as pl
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
nb=json.loads((Path(__file__).resolve().parents[1]/'references/NB0036.ipynb').read_text())
source=''.join(nb['cells'][8]['source']).split('def extract_time_features')[0]
import pandas as pd
ns={'pd':pd};exec(source,ns);orig=ns['convert_mixed_time_polars']
pattern=r'^([+-]?\d+)(?:\.([+-]?\d+))?:([+-]?\d+):([+-]?\d+)$'
def vector(col):
    value=col.str.strip_chars()
    first=value.str.extract(pattern,1).cast(pl.Int64,strict=False)
    second=value.str.extract(pattern,2).cast(pl.Int64,strict=False)
    minute=value.str.extract(pattern,3).cast(pl.Int64,strict=False)
    sec=value.str.extract(pattern,4).cast(pl.Int64,strict=False)
    return (pl.when(second.is_not_null()).then(first*86400+second*3600).otherwise(first*3600)+minute*60+sec).cast(pl.Float64)
for name in ('train','test'):
    src=pl.scan_parquet(a.data/f'{name}.parquet'); cols=[x for x in src.collect_schema() if 'duration' in x]
    for col in cols:
        vals=src.select(pl.col(col).unique().cast(pl.String)).collect()
        vals=vals.with_columns(pl.col(col).map_elements(lambda x: float(orig(x)) if orig(x) is not None else None,return_dtype=pl.Float64).alias('original'),vector(pl.col(col)).alias('vector'))
        bad=vals.filter(~pl.col('original').eq_missing(pl.col('vector')))
        if bad.height:
            print(name,col,'mismatches',bad.head(3));sys.exit(1)
        print(name,col,'unique',vals.height,'equal',flush=True)
