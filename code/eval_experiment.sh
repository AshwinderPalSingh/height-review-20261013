#!/bin/bash
#   bash eval_experiment.sh <condition> <seed>
set -e
cd /workspace/HEIGHT
NAME="exp_$1_s$2"; OUT="data/${NAME}"
CKPT=$(ls -t ${OUT}/checkpoints/*.pt | grep -v _checkpoint | head -1 | xargs basename)
echo "=== evaluating $NAME with $CKPT ==="
tmux new-session -d -s "ev_${NAME}" -c /workspace/HEIGHT \
  "bash -lc 'python3 test.py --model_dir ${OUT} --test_model ${CKPT} 2>&1 | tee logs/eval_${NAME}.log; exec bash'"
echo "launched. watch: tail -f /workspace/HEIGHT/logs/eval_${NAME}.log"
