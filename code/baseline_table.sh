#!/bin/bash
# Evaluate HEIGHT's OTHER released baselines in OUR environment, at OUR n.
#
#   tmux new-session -d -s baselines -c /workspace/HEIGHT \
#     "bash -l Experiments/baseline_table.sh 2>&1 | tee logs/BASELINES.log; exec bash"
#
# WHY. Published numbers from PaS (0.91/oracle 0.97), HICN-AS (0.94) and CrowdNav (~0.99)
# are NOT comparable to ours -- different arenas, pedestrian counts, reciprocity and
# metrics. Putting them in a table next to 88% would be misleading, and it is the same
# error that produced our retracted "19x fewer timeouts" claim (HEIGHT's published Table
# II timeout column does not reproduce from their own checkpoint).
#
# The defensible alternative: run THEIR released checkpoints in OUR config, same code,
# same n=3500. HEIGHT's Drive folder ships these already trained on the SAME random_env
# we use. That yields a baseline table where every row shares one environment.
#
# EXPECTATION, stated up front so the result is not read as a surprise: HEIGHT's paper
# reports these baselines BELOW itself (HomoGAT 0.86, DS-RNN 0.80, DRL-VO 0.50,
# A*+CNN 0.47, ORCA 0.26, DWA 0.11). We expect to beat those and to TIE HEIGHT. The
# scientific value is the CLUSTERING -- if every learned method lands in an 85-88 pp band
# while an oracle reaches 89.7%, benchmark saturation is measured, not argued.
cd /workspace/HEIGHT
mkdir -p logs
step() { echo; echo "############ $(date -u +%H:%M) :: $* ############"; }
PUB="/workspace/height_pretrained/random_env (for sim benchmarking)"

# name:zipfile  -- policy-based baselines only. ORCA and DWA are classical planners run
# through a different code path (--dwa etc.) and are handled separately, not here.
for ENTRY in DS_RNN:DS_RNN HomoGAT:HomoGAT HEIGHT_RH:HEIGHT_RH HEIGHT_HH:HEIGHT_HH HEIGHT_noAttn:HEIGHT_noAttn; do
  NAME="${ENTRY%%:*}"; ZIP="${ENTRY##*:}"
  DST="data/base_${NAME}"
  step "staging $NAME"

  if [ ! -d "${PUB}/${ZIP}" ]; then
    (cd "$PUB" && unzip -o -q "${ZIP}.zip") || { echo "SKIP $NAME: no ${ZIP}.zip"; continue; }
  fi
  CK=$(find "${PUB}/${ZIP}" -name "*.pt" 2>/dev/null | grep -v _checkpoint | head -1)
  CFG=$(find "${PUB}/${ZIP}" -name "config.py" 2>/dev/null | head -1)
  if [ -z "$CK" ] || [ -z "$CFG" ]; then echo "SKIP $NAME: checkpoint or config missing"; continue; fi

  rm -rf "$DST"; mkdir -p "$DST/checkpoints" "$DST/configs"
  cp "$CK" "$DST/checkpoints/"; cp "$CFG" "$DST/configs/config.py"
  CKN=$(basename "$CK")
  python3 set_config_value.py "$DST/configs/config.py" env.test_size 3500
  python3 set_config_value.py "$DST/configs/config.py" training.output_dir "'${DST}'"

  # preflight_BASELINE, not preflight_heightpub: the latter asserts HEIGHT's exact
  # architecture (spatial_attn / robot_linear key names) and that the added stack is off.
  # DS-RNN and HomoGAT are DIFFERENT architectures and would be rejected for reasons
  # unrelated to whether they can be evaluated. The baseline version checks only what a
  # static check can establish -- config import, test_size, environment parity -- and
  # leaves architecture compatibility to the smoke test, where load_state_dict decides.
  python3 preflight_baseline.py "$DST" "$CKN" 3500 || { echo "SKIP $NAME: preflight failed"; continue; }

  # 2-episode smoke test -- proves load_state_dict(strict=True) actually accepts their
  # weights into our Policy before spending 1.5h on it.
  python3 set_config_value.py "$DST/configs/config.py" env.test_size 2
  if python3 -u test.py --model_dir "$DST" --test_model "$CKN" > "logs/smoke_${NAME}.log" 2>&1 \
     && grep -q "has success rate" "logs/smoke_${NAME}.log"; then
    echo "SMOKE PASSED: $NAME"
  else
    echo "SKIP $NAME: smoke failed --"; tail -12 "logs/smoke_${NAME}.log"; continue
  fi
  python3 set_config_value.py "$DST/configs/config.py" env.test_size 3500

  step "EVAL $NAME (n=3500)"
  python3 -u test.py --model_dir "$DST" --test_model "$CKN" 2>&1 \
    | tee "logs/base_${NAME}.log" || echo "FAILED: $NAME"
  rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P >/dev/null 2>&1 || true
done

step "BASELINE TABLE (all n=3500, our environment, our codebase)"
printf '%-16s %s\n' "HEIGHT" "88.03%  (their released checkpoint)"
printf '%-16s %s\n' "OURS(nospeak)" "88.08 +/- 0.49%  (3 training seeds)"
for n in DS_RNN HomoGAT HEIGHT_RH HEIGHT_HH HEIGHT_noAttn; do
  printf '%-16s ' "$n"
  grep -h "has success rate" "logs/base_${n}.log" 2>/dev/null | tail -1 | sed 's/.*INFO: //' || echo "(missing)"
done
rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P || true
