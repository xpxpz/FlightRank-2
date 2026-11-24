"""Vectorized replacement for NB0036's duration parser."""
import polars as pl
PATTERN=r'^([+-]?\d+)(?:\.([+-]?\d+))?:([+-]?\d+):([+-]?\d+)$'
def duration_seconds(col):
    value=col.cast(pl.String).str.strip_chars()
    first=value.str.extract(PATTERN,1).cast(pl.Int64,strict=False)
    second=value.str.extract(PATTERN,2).cast(pl.Int64,strict=False)
    minute=value.str.extract(PATTERN,3).cast(pl.Int64,strict=False)
    second_part=value.str.extract(PATTERN,4).cast(pl.Int64,strict=False)
    return (pl.when(second.is_not_null()).then(first*86400+second*3600).otherwise(first*3600)+minute*60+second_part).cast(pl.Float64)
def convert_duration_to_seconds(df):
    cols=[c for c in df.columns if 'duration' in c]
    print('Vectorized duration conversion:',len(cols),'columns',flush=True)
    return df.with_columns([duration_seconds(pl.col(c)).alias(f'{c}_seconds') for c in cols]).drop(cols)
