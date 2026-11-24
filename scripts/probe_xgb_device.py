"""Require actual CUDA use for a small ranking workload; reject CPU fallback."""
import json,sys,warnings,time
import numpy as np
import xgboost as xgb
print('xgboost',xgb.__version__,'build',xgb.build_info(),flush=True)
if not xgb.build_info().get('USE_CUDA'):
    raise RuntimeError('XGBoost was not built with CUDA')
rng=np.random.default_rng(42); n=100000
x=rng.normal(size=(n,24)).astype('float32'); labels=np.zeros(n,dtype='float32');labels[::20]=1
d=xgb.DMatrix(x,label=labels,group=np.full(n//20,20))
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always')
    start=time.monotonic()
    m=xgb.train({'objective':'rank:pairwise','tree_method':'hist','device':'cuda:0','max_depth':8,'lambdarank_num_pair_per_sample':8},d,num_boost_round=30,evals=[(d,'train')],verbose_eval=10)
    config=json.loads(m.save_config()); device=config['learner']['generic_param']['device']
    print('device',device,'seconds',time.monotonic()-start,'warnings',[str(w.message) for w in ws],flush=True)
    if not device.startswith('cuda') or any('no visible gpu' in str(w.message).lower() or 'not compiled with cuda' in str(w.message).lower() for w in ws):
        raise RuntimeError('GPU test fell back to CPU')
