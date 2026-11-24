"""Outer-fold label-isolated behavioral aggregates with compact fold assignment.

Unlabeled feature creation is still the author's and can be transductive;
this function addresses the outer-validation label path only.
"""
import gc
import numpy as np
import polars as pl


def _group_kfold(counts, n_splits, name):
    """Match sklearn 1.9 unshuffled GroupKFold using sorted group counts."""
    counts = counts.sort('ranker_id')
    if counts.height < n_splits:
        raise ValueError('not enough groups for GroupKFold')
    weights = counts['len'].to_numpy()
    order = np.argsort(weights, kind='stable')[::-1]
    totals = np.zeros(n_splits)
    assigned = np.empty(len(weights), dtype=np.int8)
    for idx in order:
        fold = np.argmin(totals)
        totals[fold] += weights[idx]
        assigned[idx] = fold
    return counts.select('ranker_id').with_columns(pl.Series(name, assigned))


def make_nested_aggregator(outer_fold, inner_splits=5, split_file=None):
    def aggregate(train_df, test_df, agg_configs):
        counts = train_df.group_by('ranker_id').len()
        if split_file:
            outer_lookup = pl.read_parquet(split_file).select('ranker_id','fold')
            if outer_lookup['ranker_id'].n_unique() != outer_lookup.height or outer_lookup.filter(~pl.col('fold').is_in([0, 1]) | pl.col('fold').is_null()).height:
                raise ValueError('split file must assign each ranker_id a unique fold 0 or 1')
            check = counts.join(outer_lookup, on='ranker_id', how='left')
            if check['fold'].null_count() or set(check['fold'].unique().to_list()) != {0, 1}:
                raise ValueError('split file must assign all train sessions and contain both folds')
        else:
            outer_lookup = _group_kfold(counts, 10, 'fold')
        inner_counts = counts.join(outer_lookup, on='ranker_id', how='left').filter(pl.col('fold') != outer_fold).drop('fold')
        inner_lookup = _group_kfold(inner_counts, inner_splits, '_inner_fold')
        raw = (train_df.join(outer_lookup, on='ranker_id', how='left', maintain_order='left')
               .join(inner_lookup, on='ranker_id', how='left', maintain_order='left'))
        aggregate_cols = {'fold', '_inner_fold', 'selected'}
        for conf in agg_configs.values():
            aggregate_cols.update(conf['group_by'])
            aggregate_cols.add(conf['agg_col'])
        narrow = raw.select(sorted(aggregate_cols))
        outer_source = narrow.filter(pl.col('fold') != outer_fold)
        def enrich(target, source, suffix=''):
            for name, conf in agg_configs.items():
                base = source.filter(conf['filter_cond']) if conf.get('filter_cond') is not None else source
                grouped = base.group_by(conf['group_by']).agg(
                    getattr(pl.col(conf['agg_col']), conf['agg_func'])().alias(name + suffix))
                target = target.join(grouped, on=conf['group_by'], how='left')
            return target
        pieces = []
        for j in range(inner_splits):
            print(f'Nested NB0036 outer fold {outer_fold}, inner fold {j}', flush=True)
            source = outer_source.filter(pl.col('_inner_fold') != j)
            target = raw.filter((pl.col('fold') != outer_fold) & (pl.col('_inner_fold') == j)).drop('_inner_fold')
            pieces.append(enrich(target, source))
            del source, target
            gc.collect()
        print(f'Nested NB0036 outer fold {outer_fold}, validation and test', flush=True)
        outer_val = raw.filter(pl.col('fold') == outer_fold).drop('_inner_fold')
        pieces.append(enrich(outer_val, outer_source))
        result = pl.concat(pieces)
        if test_df is not None:
            test_df = enrich(test_df, outer_source, suffix=f'_fold{outer_fold}')
        return result, test_df
    return aggregate
