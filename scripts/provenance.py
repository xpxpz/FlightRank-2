"""Capture input and runtime identities without exposing data contents."""
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import sklearn
import xgboost as xgb

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def capture(data):
    return {'inputs':{name:{'sha256':digest(Path(data)/f'{name}.parquet'),
                            'bytes':(Path(data)/f'{name}.parquet').stat().st_size}
                      for name in ('train','test')},
            'versions':{'python':platform.python_version(),'numpy':np.__version__,
                        'pandas':pd.__version__,'polars':pl.__version__,
                        'scikit_learn':sklearn.__version__,'xgboost':xgb.__version__}}
