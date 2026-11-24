"""Plot recorded XGBoost NDCG@3 curves from completed runs."""
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

runs = [
    ('Nested baseline', 'results/metrics/baseline.log'),
    ('Nested NB0036', 'results/metrics/nb_nested.log'),
    ('Time split NB0036', 'results/metrics/nb_time.log'),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for ax, (title, path) in zip(axes, runs):
    records = re.findall(r'\[(\d+)\]\s+train-ndcg@3:([\d.]+)\s+val-ndcg@3:([\d.]+)', Path(path).read_text(errors='replace'))
    if len(records) < 3:
        raise ValueError(f'Insufficient metric records in {path}')
    rounds, train, valid = zip(*[(int(r), float(t), float(v)) for r, t, v in records])
    ax.plot(rounds, train, label='Train')
    ax.plot(rounds, valid, label='Validation')
    ax.set(title=title, xlabel='Boosting round', ylabel='NDCG@3')
    ax.grid(alpha=.25)
    ax.legend()
output = Path('results/figures/ndcg_curves.png')
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180)
