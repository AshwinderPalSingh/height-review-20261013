#!/bin/bash
set -euo pipefail
# CBF variants on the FROZEN checkpoint: obstacle awareness + a horizon/margin sweep.
#
#   tmux new-session -d -s cbfsweep -c /workspace/HEIGHT \
#     "bash -l /workspace/HEIGHT/Experiments/cbf_sweep.sh 2>&1 | tee /workspace/HEIGHT/logs/CBFSWEEP_reproduction_20261013.log; exec bash"
#
# WHY. The n=3500 2x2 showed the shield is null on BOTH architectures (ours -0.03 pp
# p=0.970; HEIGHT -0.46 pp p=0.559). Two specific, measured reasons it might be
# underpowered rather than useless:
#
#   1. It trades collision TYPES. At n=1000 it prevented 13 human collisions and CAUSED
#      10 obstacle collisions, because it steered away from pedestrians without scoring
#      static rectangles. `obstacles=True` closes that.
#   2. Horizon 10 = 1.0 s. Head-on closing speed is 1.0 m/s and braking from v_max needs
#      a 1.20 m gap, i.e. ~1.2 s of lookahead -- so the shield was ~2 steps short of the
#      kinematic requirement. (An earlier note claimed 2.4 s; that divided 1.20 m by the
#      HUMAN speed instead of the CLOSING speed and is wrong.)
#
# EXPECTED VALUE, stated honestly: the obstacle fix targets ~10 collisions in 3500
# episodes, about +0.3 pp -- inside noise. This is run to CLOSE the question, not because
# a publishable gain is likely. n=1000 is used for screening; anything promising gets
# re-run at 3500 before it goes near a paper.
#
# 13 Oct 2026: only the `ctrl` cell of this sweep ever ran (CBFSWEEP.log, shield off,
# stopped at episode 328), so no obstacle-aware or horizon result exists. Reproduction
# cells use isolated derived configs and preserve the original snapshot. The historical
# launcher is kept in
# Experiments/archive_pre_20261013/.
cd /workspace/HEIGHT
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOG_DIR="logs/cbf_sweep_reproduction_${RUN_ID}"
REPRO_ROOT="reproductions/cbf_sweep_${RUN_ID}"
mkdir -p "$LOG_DIR"
step() { echo; echo "############ $(date -u +%H:%M) :: $* ############"; }
SOURCE_DIR=data/exp_nospeak_s1
CK=59522.pt
N=1000

cell() {
  local TAG=$1 ENABLED=$2 HOR=$3 MAR=$4 OBS=$5
  local DIR="$REPRO_ROOT/$TAG"
  {
  step "$TAG  (source_model_dir=$SOURCE_DIR derived_model_dir=$DIR enabled=$ENABLED horizon=$HOR margin=$MAR obstacles=$OBS)"
  python3 prepare_reproduction_run.py "$SOURCE_DIR" "$DIR" "$CK"
  python3 set_config_value.py "$DIR/configs/config.py" training.output_dir "'$DIR'"
  python3 set_config_value.py "$DIR/configs/config.py" env.test_size "$N"
  python3 set_config_value.py "$DIR/configs/config.py" env.traj_dump False
  python3 set_config_value.py "$DIR/configs/config.py" cbf.enabled "$ENABLED"
  python3 set_config_value.py "$DIR/configs/config.py" cbf.horizon "$HOR"
  python3 set_config_value.py "$DIR/configs/config.py" cbf.margin "$MAR"
  python3 set_config_value.py "$DIR/configs/config.py" cbf.obstacles "$OBS"

  # Assert the flags took. A silent failure would run the same cell repeatedly and the
  # sweep would be several copies of one condition.
  python3 - "$DIR" "$ENABLED" "$HOR" "$MAR" "$OBS" "$N" <<'PYEOF'
import importlib, sys
m = sys.argv[1].replace('/', '.') + '.configs.config'
c = getattr(importlib.import_module(m), 'Config')()
assert c.training.output_dir == sys.argv[1], 'outputs must use the derived model directory'
cbf = c.cbf
assert all(hasattr(cbf, key) for key in ('enabled', 'horizon', 'margin', 'obstacles'))
assert type(cbf.enabled) is bool and type(cbf.obstacles) is bool, 'shield flags must be explicit booleans'
got = (cbf.enabled, cbf.horizon, cbf.margin, cbf.obstacles, c.env.test_size, c.env.traj_dump)
want = (sys.argv[2] == 'True', int(sys.argv[3]), float(sys.argv[4]),
        sys.argv[5] == 'True', int(sys.argv[6]), False)
print('  verify: enabled=%s horizon=%s margin=%s obstacles=%s test_size=%s traj_dump=%s env.seed=%s'
      % (got + (getattr(c.env, 'seed', '?'),)))
assert got == want, 'config did not take: got %s want %s' % (got, want)
PYEOF
  echo "  provenance:"
  sha256sum cbf_filter.py training/evaluation.py "$SOURCE_DIR/configs/config.py" "$DIR/configs/config_original.py" "$DIR/configs/config.py" "$DIR/checkpoints/$CK"

  python3 -u test.py --model_dir "$DIR" --test_model "$CK"
  python3 prepare_reproduction_run.py "$DIR" --verify-original
  } 2>&1 | tee "$LOG_DIR/sw_${TAG}.log"
  if [ "$ENABLED" = "True" ] && ! grep -q "CBF shield ACTIVE: .*obstacles=$OBS" "$LOG_DIR/sw_${TAG}.log"; then
    echo "!! $TAG: requested shield variant banner missing -- invalid cell"
    return 1
  fi
  rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P >/dev/null 2>&1 || true
}

# control, re-run at this n so every row shares a sample size
cell ctrl        False 10 0.10 False
# the obstacle fix, at the horizon we already tested
cell obs_h10     True  10 0.10 True
# horizon sweep, obstacle-aware throughout (12 = the kinematic requirement)
cell obs_h12     True  12 0.10 True
cell obs_h15     True  15 0.10 True
cell obs_h20     True  20 0.10 True
# a wider margin at the kinematic horizon: fires earlier still
cell obs_h12_m25 True  12 0.25 True

# Each cell edits only its own derived config; original snapshots need no reset.

step "SWEEP REPRODUCTION RESULTS (n=$N, frozen checkpoint $CK, logs=$LOG_DIR)"
printf '%-14s %-46s %s\n' "cell" "outcome" "override"
for t in ctrl obs_h10 obs_h12 obs_h15 obs_h20 obs_h12_m25; do
  R=$(grep -h "has success rate" "$LOG_DIR/sw_${t}.log" 2>/dev/null | tail -1 | sed 's/.*INFO: //')
  O=$(grep -h "CBF: " "$LOG_DIR/sw_${t}.log" 2>/dev/null | tail -1 | sed 's/.*INFO: CBF: //')
  printf '%-14s %-46s %s\n' "$t" "${R:-(missing)}" "${O:-—}"
done
echo
echo "success COUNTS (for the significance test):"
for t in ctrl obs_h10 obs_h12 obs_h15 obs_h20 obs_h12_m25; do
  printf '  %-14s ' "$t"
  grep -h "nav time (successes)" "$LOG_DIR/sw_${t}.log" 2>/dev/null | tail -1 | sed 's/.*over //' || echo "(missing)"
done
echo
echo "READ: compare each shielded cell against ctrl. A cell that raises success WITHOUT"
echo "  raising timeout, and whose override rate is non-zero, is worth an n=3500 re-run."
echo "  Anything under ~2 pp will not clear significance at n=3500 x 3 seeds."
rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P || true
