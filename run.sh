#!/usr/bin/env bash
set -euo pipefail

export USE_QUANTIZATION=false
export QUANTIZATION_BITS=16
export BATCH_SIZE=128
MODEL=model1

run_pred () {
  local track=$1
  local run=$2
  (
    cd "$track"
    bash pred.sh "$run" "$MODEL"
  )
}

export -f run_pred

for track in classification-track generation-track multilingual-track; do
  for run in 1 2; do
    run_pred "$track" "$run" &
  done
done

wait
echo "✅ All predictions finished"
