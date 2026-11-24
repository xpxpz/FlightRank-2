"""Official FlightRank HitRate@3 and group integrity checks."""
import numpy as np


def hitrate_at_3(groups, labels, scores):
    groups, labels, scores = map(np.asarray, (groups, labels, scores))
    if not (len(groups) == len(labels) == len(scores)):
        raise ValueError('groups, labels and scores must have the same length')
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('scores must be finite and labels binary')
    order = np.argsort(groups, kind='stable')
    g, y, s = groups[order], labels[order], scores[order]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1, len(g)]
    hits = []
    for a, b in zip(starts[:-1], starts[1:]):
        if y[a:b].sum() != 1:
            raise ValueError(f'group {g[a]!r} does not contain exactly one positive')
        if b-a > 10:
            top = np.argsort(-s[a:b], kind='stable')[:3]
            hits.append(int(y[a:b][top].max()))
    if not hits:
        raise ValueError('no groups with more than 10 candidates')
    return float(np.mean(hits))
