#!/bin/bash
set -euo pipefail
# The 2x2 at n=3500: {HEIGHT, ours} x {shield off, shield on}.
#
#   tmux new-session -d -s cbf2x2 -c /workspace/HEIGHT \
#     "bash -l Experiments/cbf_2x2.sh 2>&1 | tee logs/CBF2X2_reproduction_20261013.log; exec bash"
#
# WHY. At n=1000 the shield HELPED our system (+0.3 pp) and HURT HEIGHT (-0.8 pp), at
# near-identical firing rates (0.52% vs 0.55%). The architecture gap widened from
# +1.30 pp unshielded to +2.40 pp shielded (p=0.094). Every one of those numbers is
# non-significant at n=1000 (CI ~ +-2.7 pp) -- a 1.1 pp interaction assembled from two
# nulls is exactly the pattern that dissolves at higher n. This run resolves it.
#
# If the +2.40 pp shielded gap survives at n=3500, p falls to ~0.01 and there is a
# significant result over HEIGHT. If it collapses, the last open question is closed and
# the saturation paper is complete. Either outcome is worth the 12 hours.
#
# All four cells are re-run even though two exist at n=1000, so every number in the
# table comes from the same sample size.
#
# 13 Oct 2026: the reported run (9-10 Sep) never set cbf.obstacles, so it relied on the
# default False and tested the pedestrian-only shield. This launcher now sets and asserts
# every shield flag, including obstacles=False, so that a config left with obstacles=True
# by Experiments/cbf_sweep.sh cannot turn a reproduction into a different experiment. It
# also prints the effective flags and the source, config and checkpoint hashes per cell.
# The launcher as it ran is kept in Experiments/archive_pre_20261013/.
cd /workspace/HEIGHT
# Reproductions never replace the original September logs.
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOG_DIR="logs/cbf_2x2_reproduction_${RUN_ID}"
REPRO_ROOT="reproductions/cbf_2x2_${RUN_ID}"
mkdir -p "$LOG_DIR"
step() { echo; echo "############ $(date -u +%H:%M) :: $* ############"; }

run_cell() {
  local TAG=$1 SOURCE_DIR=$2 CK=$3 CBF=$4
  local DIR="$REPRO_ROOT/$TAG"
  {
  step "$TAG  (source_model_dir=$SOURCE_DIR derived_model_dir=$DIR cbf=$CBF)"
  python3 prepare_reproduction_run.py "$SOURCE_DIR" "$DIR" "$CK"
  python3 set_config_value.py "$DIR/configs/config.py" training.output_dir "'$DIR'"
  python3 set_config_value.py "$DIR/configs/config.py" env.test_size 3500
  python3 set_config_value.py "$DIR/configs/config.py" cbf.enabled "$CBF"
  python3 set_config_value.py "$DIR/configs/config.py" cbf.horizon 10
  python3 set_config_value.py "$DIR/configs/config.py" cbf.margin 0.10
  python3 set_config_value.py "$DIR/configs/config.py" cbf.obstacles False
  # Prove every flag took: a silent failure here would run the wrong cell, and a stale
  # obstacles=True would test a different shield from the one reported.
  python3 - "$DIR" "$CBF" <<'PYEOF'
import importlib, sys
m = sys.argv[1].replace('/', '.') + '.configs.config'
c = getattr(importlib.import_module(m), 'Config')()
assert c.training.output_dir == sys.argv[1], 'outputs must use the derived model directory'
cbf = c.cbf
assert all(hasattr(cbf, key) for key in ('enabled', 'horizon', 'margin', 'obstacles'))
assert type(cbf.enabled) is bool and type(cbf.obstacles) is bool, 'shield flags must be explicit booleans'
got = (cbf.enabled, cbf.horizon, cbf.margin, cbf.obstacles, c.env.test_size)
want = (sys.argv[2] == 'True', 10, 0.10, False, 3500)
print('  verify: enabled=%s horizon=%s margin=%s obstacles=%s test_size=%s  env.seed=%s'
      % (got + (getattr(c.env, 'seed', '?'),)))
assert got == want, 'shield config did not take: got %s want %s -- ABORT this cell' % (got, want)
PYEOF
  echo "  provenance:"
  sha256sum cbf_filter.py training/evaluation.py "$SOURCE_DIR/configs/config.py" "$DIR/configs/config_original.py" "$DIR/configs/config.py" "$DIR/checkpoints/$CK" | sed 's/^/    /'

  python3 -u test.py --model_dir "$DIR" --test_model "$CK"
  python3 prepare_reproduction_run.py "$DIR" --verify-original
  } 2>&1 | tee "$LOG_DIR/x_${TAG}.log"
  # A shielded cell that never logs the banner is really an unshielded cell, and a banner
  # without obstacles=False is not the reported variant.
  if [ "$CBF" = "True" ] && ! grep -q "CBF shield ACTIVE: .*obstacles=False" "$LOG_DIR/x_${TAG}.log"; then
    echo "!! $TAG: pedestrian-only shield was requested but its banner is missing -- invalid cell"
    return 1
  fi
  rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P >/dev/null 2>&1 || true
}

run_cell height_off   data/exp_heightpub  237800.pt False
run_cell height_on    data/exp_heightpub  237800.pt True
run_cell ours_off     data/exp_nospeak_s1 59522.pt  False
run_cell ours_on      data/exp_nospeak_s1 59522.pt  True

# Each cell edits only its own derived config; original snapshots need no reset.

step "2x2 REPRODUCTION RESULTS (n=3500, logs=$LOG_DIR)"
for t in height_off height_on ours_off ours_on; do
  printf '%-12s ' "$t"
  grep -h "has success rate" "$LOG_DIR/x_${t}.log" 2>/dev/null | tail -1 | sed 's/.*INFO: //' || echo "(missing)"
done
echo
echo "success COUNTS (for the significance test):"
for t in height_off height_on ours_off ours_on; do
  printf '%-12s ' "$t"
  grep -h "nav time (successes)" "$LOG_DIR/x_${t}.log" 2>/dev/null | tail -1 | sed 's/.*over //' || echo "(missing)"
done
echo
echo "override rates:"
for t in height_on ours_on; do
  printf '%-12s ' "$t"
  grep -h "CBF: " "$LOG_DIR/x_${t}.log" 2>/dev/null | tail -1 | sed 's/.*INFO: //' || echo "(missing)"
done
rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P || true
