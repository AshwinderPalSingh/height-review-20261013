#!/bin/bash
# TIER 2 for the journal paper: bring every curve to uniform n=3500.
#
#   tmux new-session -d -s tier2 -c /workspace/HEIGHT \
#     "bash -l /workspace/HEIGHT/Experiments/tier2_chain.sh 2>&1 | tee /workspace/HEIGHT/logs/TIER2.log; exec bash"
#
# STAGE A  reciprocity middle points  vp in {0.4, 0.6, 0.8} x {ours, height}   6 cells ~16 h
#   WHY: the paper's central figure currently has its two endpoints at n=3500 and the
#   four middle points at n=1000. A referee will notice the headline curve is the least
#   uniformly powered thing in the paper. Endpoints already done (re_vp02_*, re_vp10_*).
#
# STAGE B  oracle-vs-occluded at {5, 8, 15} m x {occl, oracle}                 6 cells ~16 h
#   WHY: Table IV has only 3 m and 25 m. Three more ranges turn it into a curve.
#   The n=1000 screening found these non-significant and non-monotone; at n=3500 either
#   a monotone trend emerges or the "flat in the middle, only 3 m matters" reading is
#   confirmed at proper power. Either is reportable. PRE-REGISTERED: we expect the gap
#   at 5 m to sit between 3 m (+3.03) and 25 m (0.00); if it is instead near zero, the
#   headroom is a very-short-range effect only.
#
# Both stages are eval-only on frozen checkpoints. Syncs to Drive after every cell.
# ~32 h total. Reuses the cell() pattern from recip_endpoints.sh / range_confirm.sh,
# both of which ran correctly at n=3500.
cd /workspace/HEIGHT
mkdir -p logs
step() { echo; echo "############ $(date -u '+%F %H:%M') :: $* ############"; }
sync_drive() { rclone copy logs/ gdrive:HEIGHT_Backup/logs/ 2>&1 | grep -vE "NOTICE|Transferred|Checks|Elapsed" || true; }
N=3500

# refuse to start on top of another job
if pgrep -f "python3 .*test.py" >/dev/null 2>&1 || pgrep -f "python3 .*train.py" >/dev/null 2>&1; then
  echo "!! another eval/train is running -- refusing to start"; pgrep -af "python3 .*(test|train).py"; exit 1
fi

# ---------------------------------------------------------------- STAGE A
recip_cell() {
  local VP=$1 POL=$2 SRC=$3 CK=$4
  local T=${VP/./}; local TAG="vp${T}_${POL}"; local DST="data/re_${TAG}"
  if grep -q "nav time (successes)" "logs/re_${TAG}.log" 2>/dev/null; then
    echo "  re_${TAG} already complete -- skipping"; return 0
  fi
  step "A: visible_prob=${VP}  policy=${POL}  n=${N}"
  rm -rf "$DST"; mkdir -p "$DST/checkpoints" "$DST/configs"
  cp "${SRC}/checkpoints/${CK}" "$DST/checkpoints/" || { echo "SKIP ${TAG}: missing ckpt"; return 1; }
  cp "${SRC}/configs/config.py" "$DST/configs/config.py" || { echo "SKIP ${TAG}: missing config"; return 1; }
  python3 set_config_value.py "$DST/configs/config.py" robot.visible_prob "$VP"
  python3 set_config_value.py "$DST/configs/config.py" env.test_size "$N"
  python3 set_config_value.py "$DST/configs/config.py" training.output_dir "'${DST}'"
  python3 set_config_value.py "$DST/configs/config.py" lidar.sensor_range 25
  python3 set_config_value.py "$DST/configs/config.py" sim.oracle_visibility False
  python3 set_config_value.py "$DST/configs/config.py" cbf.enabled False
  python3 set_config_value.py "$DST/configs/config.py" env.traj_dump False
  python3 - "$DST" "$VP" "$N" "$POL" <<'PYEOF'
import importlib, sys
c = getattr(importlib.import_module(sys.argv[1].replace('/', '.') + '.configs.config'), 'Config')()
vp, n, pol, pm = float(c.robot.visible_prob), int(c.env.test_size), sys.argv[4], c.sim.predict_method
print('  verify: visible_prob=%s test_size=%d predict=%s range=%s oracle=%s cbf=%s'
      % (vp, n, pm, c.lidar.sensor_range, getattr(c.sim,'oracle_visibility',None), getattr(getattr(c,'cbf',None),'enabled',None)))
assert vp == float(sys.argv[2]) and n == int(sys.argv[3]) and float(c.lidar.sensor_range) == 25.0
assert (pol == 'ours' and pm == 'truth') or (pol == 'height' and pm == 'none'), 'predict_method mismatch for %s: %s' % (pol, pm)
PYEOF
  [ $? -ne 0 ] && { echo "SKIP ${TAG}: verification failed"; return 1; }
  python3 -u test.py --model_dir "$DST" --test_model "$CK" 2>&1 | tee "logs/re_${TAG}.log"
  sync_drive
}
for VP in 0.4 0.6 0.8; do
  recip_cell "$VP" ours   data/exp_nospeak_s1 59522.pt
  recip_cell "$VP" height data/exp_heightpub  237800.pt
done

# ---------------------------------------------------------------- STAGE B
range_cell() {
  local RANGE=$1 ORACLE=$2
  local TAG="rc${RANGE}_$( [ "$ORACLE" = "True" ] && echo oracle || echo occl )"; local DST="data/rc_${TAG}"
  if grep -q "nav time (successes)" "logs/rc_${TAG}.log" 2>/dev/null; then
    echo "  rc_${TAG} already complete -- skipping"; return 0
  fi
  step "B: range=${RANGE} m  oracle=${ORACLE}  n=${N}"
  rm -rf "$DST"; mkdir -p "$DST/checkpoints" "$DST/configs"
  cp data/exp_nospeak_s1/checkpoints/59522.pt "$DST/checkpoints/" || { echo "SKIP ${TAG}: missing ckpt"; return 1; }
  cp data/exp_nospeak_s1/configs/config.py "$DST/configs/config.py" || { echo "SKIP ${TAG}: missing config"; return 1; }
  python3 set_config_value.py "$DST/configs/config.py" lidar.sensor_range "$RANGE"
  python3 set_config_value.py "$DST/configs/config.py" env.test_size "$N"
  python3 set_config_value.py "$DST/configs/config.py" sim.oracle_visibility "$ORACLE"
  python3 set_config_value.py "$DST/configs/config.py" training.output_dir "'${DST}'"
  python3 set_config_value.py "$DST/configs/config.py" robot.visible_prob 0.2
  python3 set_config_value.py "$DST/configs/config.py" cbf.enabled False
  python3 set_config_value.py "$DST/configs/config.py" env.traj_dump False
  python3 - "$DST" "$RANGE" "$ORACLE" "$N" <<'PYEOF'
import importlib, sys
c = getattr(importlib.import_module(sys.argv[1].replace('/', '.') + '.configs.config'), 'Config')()
r, o, n = float(c.lidar.sensor_range), str(getattr(c.sim,'oracle_visibility',False)), int(c.env.test_size)
print('  verify: sensor_range=%s oracle=%s test_size=%d visible_prob=%s cbf=%s' % (r, o, n, c.robot.visible_prob, getattr(getattr(c,'cbf',None),'enabled',None)))
assert r == float(sys.argv[2]) and o == sys.argv[3] and n == int(sys.argv[4])
assert float(c.robot.visible_prob) == 0.2, 'reciprocity must be the DEFAULT for the range curve'
PYEOF
  [ $? -ne 0 ] && { echo "SKIP ${TAG}: verification failed"; return 1; }
  python3 -u test.py --model_dir "$DST" --test_model 59522.pt 2>&1 | tee "logs/rc_${TAG}.log"
  sync_drive
}
for R in 5 8 15; do
  range_cell "$R" False
  range_cell "$R" True
done

# ---------------------------------------------------------------- REPORT
step "TIER 2 COMPLETE"
echo "--- A. reciprocity, all six levels at n=$N ---"
printf '  %-8s %-9s %-9s\n' "vp" "ours" "height"
for VP in 0.0 0.2 0.4 0.6 0.8 1.0; do
  T=${VP/./}
  O=$(grep -h "nav time (successes)" logs/re_vp${T}_ours.log   2>/dev/null | tail -1 | sed 's/.*over n=//')
  H=$(grep -h "nav time (successes)" logs/re_vp${T}_height.log 2>/dev/null | tail -1 | sed 's/.*over n=//')
  printf '  %-8s %-9s %-9s\n' "$VP" "${O:-(n=1000 only)}" "${H:-(n=1000 only)}"
done
echo "  (vp=0.0 was not re-run: it is off the benchmark's range and the n=1000 value stands as screening)"
echo
echo "--- B. oracle headroom vs range at n=$N ---"
printf '  %-6s %-9s %-9s\n' "range" "occl" "oracle"
for R in 3 5 8 15 25; do
  O=$(grep -h "nav time (successes)" logs/rc_rc${R}_occl.log   2>/dev/null | tail -1 | sed 's/.*over n=//')
  A=$(grep -h "nav time (successes)" logs/rc_rc${R}_oracle.log 2>/dev/null | tail -1 | sed 's/.*over n=//')
  printf '  %-6s %-9s %-9s\n' "${R} m" "${O:-(missing)}" "${A:-(missing)}"
done
sync_drive
echo "Chain finished $(date -u '+%F %H:%M') UTC."
