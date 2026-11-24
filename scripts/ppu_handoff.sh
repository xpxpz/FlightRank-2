#!/usr/bin/env bash
# Finish in-progress CPU fold 3, then run remaining work on the verified PPU.
set -Eeuo pipefail
: "${RUN_ROOT:?}" "${CPU_ROOT:?}" "${DATA_DIR:?}" "${PROJECT_DIR:?}" "${BASELINE_DIR:?}" "${CPU_PARENT:?}" "${CPU_WORKER:?}"
cd "$PROJECT_DIR"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/reports" "$RUN_ROOT/models"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2
PYTHON="$PROJECT_DIR/.venv_gpu/bin/python"
status="$RUN_ROOT/status.txt"; log="$RUN_ROOT/pipeline.log"; stage=wait_cpu_fold_3
trap 'rc=$?; printf "%s FAILED stage=%s exit=%s\n" "$(date -u +%FT%TZ)" "$stage" "$rc" | tee -a "$status" "$log"' ERR
run() {
  stage=$1; shift
  printf '%s START %s\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$status" "$log"
  python3 scripts/run_logged.py --log-dir "$RUN_ROOT/logs/$stage" --interval 1 --cgroup-limit-gib 160 --ppu-smi /usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi -- "$@" >> "$log" 2>&1
  printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$status" "$log"
}
printf '%s WAIT CPU fold 3\n' "$(date -u +%FT%TZ)" | tee -a "$status" "$log"
while kill -0 "$CPU_WORKER" 2>/dev/null; do sleep 30; done
# Parent is deliberately paused, so the child wrapper may be a zombie. Wait
# for its final metadata and require a successful result before handoff.
for attempt in $(seq 1 12); do
  if "$PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); sys.exit(0 if x.get("exit_code")==0 else 1)' "$CPU_ROOT/logs/nb_fold_3_resume/run.json" 2>/dev/null && test -s "$CPU_ROOT/models/nb0036/fold_3/result.json"; then break; fi
  sleep 5
done
"$PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x.get("exit_code")==0' "$CPU_ROOT/logs/nb_fold_3_resume/run.json"
test -s "$CPU_ROOT/models/nb0036/fold_3/result.json"
# Stop only the paused queue shell after its fold worker is finished. It would
# otherwise start duplicate CPU fold 4; completed results are left untouched.
kill -KILL "$CPU_PARENT"
printf '%s COMPLETE CPU fold 3; switched to PPU queue\n' "$(date -u +%FT%TZ)" | tee -a "$status" "$log"
run ppu_preflight "$PYTHON" scripts/probe_xgb_device.py
for fold in 4 5 6 7 8 9; do
  run "nb_fold_${fold}_ppu" "$PYTHON" scripts/run_nb0036_ppu.py --data "$DATA_DIR" --out "$CPU_ROOT/models/nb0036" --folds "$fold" --duration-mode vectorized --compact-aggregates --test-aggregates requested --rounds 3500 --threads 12 --device cuda:0
done
run combine_nb "$PYTHON" scripts/combine_folds.py --fold-root "$CPU_ROOT/models/nb0036" --test "$DATA_DIR/test.parquet" --output "$RUN_ROOT/models/nb0036_combined" --allow-mixed-device
run baseline_original_ppu "$PYTHON" scripts/run_baseline_ppu.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_original" --split original --rounds 1000 --threads 12 --device cuda:0
run baseline_nested_ppu "$PYTHON" scripts/run_baseline_ppu.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_nested" --carrier-stats nested --fold 0 --rounds 1000 --threads 12 --device cuda:0
for variant in none price convenience behavior; do
  run "nb_nested_${variant}_ppu" "$PYTHON" scripts/run_nb0036_ppu.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb_nested_$variant" --aggregate-mode nested --duration-mode vectorized --ablate "$variant" --folds 0 --rounds 3500 --threads 12 --device cuda:0
  run "compare_nested_$variant" "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/baseline_nested/validation.parquet" --candidate "$RUN_ROOT/models/nb_nested_$variant/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_nested_$variant.json"
  if [[ "$variant" != none ]]; then
    run "ablation_vs_full_$variant" "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/nb_nested_none/fold_0/validation.parquet" --candidate "$RUN_ROOT/models/nb_nested_$variant/fold_0/validation.parquet" --output "$RUN_ROOT/reports/ablation_vs_full_$variant.json"
  fi
done
run time_split "$PYTHON" scripts/make_time_split.py --train "$DATA_DIR/train.parquet" --output "$RUN_ROOT/time_split.parquet"
run baseline_time_ppu "$PYTHON" scripts/run_baseline_ppu.py --data "$DATA_DIR" --out "$RUN_ROOT/models/baseline_time" --carrier-stats nested --split-file "$RUN_ROOT/time_split.parquet" --rounds 1000 --threads 12 --device cuda:0
run nb_time_ppu "$PYTHON" scripts/run_nb0036_ppu.py --data "$DATA_DIR" --out "$RUN_ROOT/models/nb_time" --aggregate-mode nested --duration-mode vectorized --folds 0 --split-file "$RUN_ROOT/time_split.parquet" --rounds 3500 --threads 12 --device cuda:0
run compare_time "$PYTHON" scripts/compare_predictions.py --baseline "$RUN_ROOT/models/baseline_time/validation.parquet" --candidate "$RUN_ROOT/models/nb_time/fold_0/validation.parquet" --output "$RUN_ROOT/reports/comparison_time.json"
stage=complete
printf '%s COMPLETE all\n' "$(date -u +%FT%TZ)" | tee -a "$status" "$log"
