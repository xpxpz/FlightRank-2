#!/usr/bin/env bash
# Serial FlightRank experiments. A failure stops the queue and leaves logs intact.
set -Eeuo pipefail
: "${RUN_ROOT:?Set RUN_ROOT to a new absolute directory}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing full parquet files}"
: "${PROJECT_DIR:?Set PROJECT_DIR to the code directory}"
MEMORY_GIB=${MEMORY_GIB:-48}
THREADS=${THREADS:-3}
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/models" "$RUN_ROOT/reports"
cd "$PROJECT_DIR"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS="$THREADS" POLARS_MAX_THREADS="$THREADS"
status="$RUN_ROOT/status.txt"
log="$RUN_ROOT/pipeline.log"
stage=initializing
trap 'rc=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date -u +%FT%TZ)" "$stage" "$rc" | tee -a "$status" "$log"' ERR
run() {
    stage=$1; shift
    printf '%s START %s\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$status" "$log"
    python3 scripts/run_logged.py --log-dir "$RUN_ROOT/logs/$stage" --memory-gib "$MEMORY_GIB" --nice 5 -- "$@" >> "$log" 2>&1
    printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$status" "$log"
}
PYTHON="$PROJECT_DIR/.venv/bin/python"
run baseline_group "$PYTHON" scripts/run_baseline.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_group" --fold 0 --rounds 1000 --threads "$THREADS"
run nb_fold_0 "$PYTHON" scripts/run_nb0036.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb0036" --folds 0 --test-aggregates requested --rounds 3500 --threads "$THREADS"
run compare_fold_0 "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/baseline_group/validation.parquet" --candidate "$RUN_ROOT/models/nb0036/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_fold_0.json"
for fold in 1 2 3 4 5 6 7 8 9; do
    run "nb_fold_$fold" "$PYTHON" scripts/run_nb0036.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb0036" --folds "$fold" --test-aggregates requested --rounds 3500 --threads "$THREADS"
done
run combine_nb "$PYTHON" scripts/combine_folds.py --fold-root "$RUN_ROOT/models/nb0036" --test "$DATA_DIR/test.parquet" --output "$RUN_ROOT/models/nb0036_combined"
# Run original baseline cutoff only when it does not split a session. This
# check is performed before launch, using inspect_data's authoritative output.
if [[ "${ORIGINAL_BOUNDARY_SAFE:-no}" == yes ]]; then
    run baseline_original "$PYTHON" scripts/run_baseline.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_original" --split original --rounds 1000 --threads "$THREADS"
else
    printf '%s SKIP baseline_original: original boundary not verified safe\n' "$(date -u +%FT%TZ)" | tee -a "$status" "$log"
fi
# Label-isolated reference and NB0036 behavior/price/convenience ablations.
run baseline_nested "$PYTHON" scripts/run_baseline.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_nested" --carrier-stats nested --fold 0 --rounds 1000 --threads "$THREADS"
for variant in none price convenience behavior; do
    run "nb_nested_$variant" "$PYTHON" scripts/run_nb0036.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb_nested_$variant" --aggregate-mode nested --ablate "$variant" --folds 0 --rounds 3500 --threads "$THREADS"
    run "compare_nested_$variant" "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/baseline_nested/validation.parquet" --candidate "$RUN_ROOT/models/nb_nested_$variant/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_nested_$variant.json"
done
run time_split "$PYTHON" scripts/make_time_split.py --train "$DATA_DIR/train.parquet" --output "$RUN_ROOT/time_split.parquet"
run baseline_time "$PYTHON" scripts/run_baseline.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_time" --carrier-stats nested --split-file "$RUN_ROOT/time_split.parquet" --rounds 1000 --threads "$THREADS"
run nb_time "$PYTHON" scripts/run_nb0036.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb_time" --aggregate-mode nested --folds 0 --split-file "$RUN_ROOT/time_split.parquet" --rounds 3500 --threads "$THREADS"
run compare_time "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/baseline_time/validation.parquet" --candidate "$RUN_ROOT/models/nb_time/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_time.json"
stage=complete
printf '%s COMPLETE all\n' "$(date -u +%FT%TZ)" | tee -a "$status" "$log"
