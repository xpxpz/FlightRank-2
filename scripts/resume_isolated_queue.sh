#!/usr/bin/env bash
set -Eeuo pipefail
: "${RUN_ROOT:?}" "${DATA_DIR:?}" "${PROJECT_DIR:?}" "${BASELINE_DIR:?}" "${BASELINE_PID:?}"
cd "$PROJECT_DIR"; mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/models" "$RUN_ROOT/reports"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2
PYTHON="$PROJECT_DIR/.venv_gpu/bin/python"; status="$RUN_ROOT/status.txt"; log="$RUN_ROOT/pipeline.log"; stage=wait_baseline
trap 'rc=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date -u +%FT%TZ)" "$stage" "$rc" | tee -a "$status" "$log"' ERR
run() {
 stage=$1; shift
 printf '%s START %s\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$status" "$log"
 python3 scripts/run_logged.py --log-dir "$RUN_ROOT/logs/$stage" --interval 1 --cgroup-limit-gib 160 --ppu-smi /usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi -- "$@" >> "$log" 2>&1
 printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$status" "$log"
}
printf '%s WAIT isolated baseline\n' "$(date -u +%FT%TZ)" | tee -a "$status" "$log"
while kill -0 "$BASELINE_PID" 2>/dev/null; do
 state=$(ps -o stat= -p "$BASELINE_PID" 2>/dev/null || true)
 [[ "$state" == Z* ]] && break
 sleep 30
done
"$PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("exit_code")==0' "$RUN_ROOT/baseline_log/run.json"
test -s "$BASELINE_DIR/result.json"
printf '%s COMPLETE isolated baseline\n' "$(date -u +%FT%TZ)" | tee -a "$status" "$log"
for variant in none price convenience behavior; do
 run "nb_nested_${variant}_ppu" "$PYTHON" scripts/run_nb0036_ppu.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb_nested_$variant" --aggregate-mode nested --duration-mode vectorized --ablate "$variant" --folds 0 --rounds 3500 --threads 12 --device cuda:0
 run "compare_nested_$variant" "$PYTHON" scripts/compare_predictions.py --baseline "$BASELINE_DIR/validation.parquet" --candidate "$RUN_ROOT/models/nb_nested_$variant/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_nested_$variant.json"
 if [[ "$variant" != none ]]; then
  run "ablation_vs_full_$variant" "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/nb_nested_none/fold_0/validation.parquet" --candidate "$RUN_ROOT/models/nb_nested_$variant/fold_0/validation.parquet" --output "$RUN_ROOT/reports/ablation_vs_full_$variant.json"
 fi
done
run time_split "$PYTHON" scripts/make_time_split.py --train "$DATA_DIR/train.parquet" --output "$RUN_ROOT/time_split.parquet"
run baseline_time_ppu "$PYTHON" scripts/run_baseline_ppu_compact.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_time" --carrier-stats nested --split-file "$RUN_ROOT/time_split.parquet" --rounds 1000 --threads 12 --device cuda:0
run nb_time_ppu "$PYTHON" scripts/run_nb0036_ppu.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb_time" --aggregate-mode nested --duration-mode vectorized --folds 0 --split-file "$RUN_ROOT/time_split.parquet" --rounds 3500 --threads 12 --device cuda:0
run compare_time "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/baseline_time/validation.parquet" --candidate "$RUN_ROOT/models/nb_time/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_time.json"
stage=complete; printf '%s COMPLETE all\n' "$(date -u +%FT%TZ)" | tee -a "$status" "$log"
