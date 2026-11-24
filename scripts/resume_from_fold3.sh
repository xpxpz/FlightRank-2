#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUN_ROOT:?}" "${DATA_DIR:?}" "${PROJECT_DIR:?}" "${BASELINE_DIR:?}"
cd "$PROJECT_DIR"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/models" "$RUN_ROOT/reports"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2
PYTHON="$PROJECT_DIR/.venv/bin/python"
status="$RUN_ROOT/status.txt"; log="$RUN_ROOT/pipeline.log"; stage=initializing
trap 'rc=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date -u +%FT%TZ)" "$stage" "$rc" | tee -a "$status" "$log"' ERR
run() {
  stage=$1; shift
  printf '%s START %s\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$status" "$log"
  python3 scripts/run_logged.py --log-dir "$RUN_ROOT/logs/$stage" --interval 1 --cgroup-limit-gib 160 -- "$@" >> "$log" 2>&1
  printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$status" "$log"
}
for fold in 3 4 5 6 7 8 9; do
  run "nb_fold_${fold}_resume" "$PYTHON" scripts/run_nb0036.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb0036" --folds "$fold" --duration-mode vectorized --compact-aggregates --test-aggregates requested --rounds 3500 --threads 12
  if [[ "$fold" == 0 ]]; then
    run compare_fold_0 "$PYTHON" scripts/compare_predictions.py --baseline "$BASELINE_DIR/validation.parquet" --candidate "$RUN_ROOT/models/nb0036/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_fold_0.json"
  fi
done
run combine_nb "$PYTHON" scripts/combine_folds.py --fold-root "$RUN_ROOT/models/nb0036" --test "$DATA_DIR/test.parquet" --output "$RUN_ROOT/models/nb0036_combined"
run baseline_original "$PYTHON" scripts/run_baseline.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_original" --split original --rounds 1000 --threads 12
run baseline_nested "$PYTHON" scripts/run_baseline.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_nested" --carrier-stats nested --fold 0 --rounds 1000 --threads 12
for variant in none price convenience behavior; do
  run "nb_nested_$variant" "$PYTHON" scripts/run_nb0036.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb_nested_$variant" --aggregate-mode nested --duration-mode vectorized --ablate "$variant" --folds 0 --rounds 3500 --threads 12
  run "compare_nested_$variant" "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/baseline_nested/validation.parquet" --candidate "$RUN_ROOT/models/nb_nested_$variant/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_nested_$variant.json"
done
run time_split "$PYTHON" scripts/make_time_split.py --train "$DATA_DIR/train.parquet" --output "$RUN_ROOT/time_split.parquet"
run baseline_time "$PYTHON" scripts/run_baseline.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_time" --carrier-stats nested --split-file "$RUN_ROOT/time_split.parquet" --rounds 1000 --threads 12
run nb_time "$PYTHON" scripts/run_nb0036.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb_time" --aggregate-mode nested --duration-mode vectorized --folds 0 --split-file "$RUN_ROOT/time_split.parquet" --rounds 3500 --threads 12
run compare_time "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/baseline_time/validation.parquet" --candidate "$RUN_ROOT/models/nb_time/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_time.json"
stage=complete
printf '%s COMPLETE all\n' "$(date -u +%FT%TZ)" | tee -a "$status" "$log"
