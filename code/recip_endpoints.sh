#!/bin/bash
# Anchor the reciprocity curve's two most-cited points at n=3500.
#
#   tmux new-session -d -s recipend -c /workspace/HEIGHT \
#     "bash -l /workspace/HEIGHT/Experiments/recip_endpoints.sh 2>&1 | tee /workspace/HEIGHT/logs/RECIPEND.log; exec bash"
#
# WHY. The dose-response curve (visible_prob 0.0 -> 1.0, both systems) is the paper's
# central result, and every one of its 12 cells is n=1000 -- screening grade -- while the
# secondary results (oracle, social metrics, 2x2 shield) are all n=3500. A referee will
# notice the headline figure is the LEAST powered thing in the paper.
#
# This re-runs only the two points that will actually be quoted: the benchmark default
# (0.2) and full reciprocity (1.0), for both systems. Four cells, ~3 h.
#
# PRE-REGISTERED, from the n=1000 sweep:
#   ours   0.2 -> 1.0 : 89.3% -> 99.1%   (+9.8 pp)
#   HEIGHT 0.2 -> 1.0 : 86.8% -> 98.7%   (+11.9 pp)
#   at 1.0 the two CONVERGE (+0.4 pp, p=0.39)
# If those hold at n=3500 the curve is anchored. If the convergence at 1.0 dissolves, the
# "architectural gap vanishes when difficulty is removed" claim must be dropped.
#
# Ours runs predict_method='truth' (oracle), HEIGHT runs 'none' -- same as the sweep.
cd /workspace/HEIGHT
mkdir -p logs
step() { echo; echo "############ $(date -u +%H:%M) :: $* ############"; }
N=3500

cell() {
  local VP=$1 POL=$2 SRC=$3 CK=$4
  local T=${VP/./}
  local TAG="vp${T}_${POL}"
  local DST="data/re_${TAG}"
  step "visible_prob=${VP}  policy=${POL}  n=${N}"

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
m = sys.argv[1].replace('/', '.') + '.configs.config'
c = getattr(importlib.import_module(m), 'Config')()
vp, n, pol = float(c.robot.visible_prob), int(c.env.test_size), sys.argv[4]
pm = c.sim.predict_method
print('  verify: visible_prob=%s test_size=%d predict=%s range=%s oracle=%s cbf=%s'
      % (vp, n, pm, c.lidar.sensor_range, getattr(c.sim,'oracle_visibility',None),
         getattr(getattr(c,'cbf',None),'enabled',None)))
assert vp == float(sys.argv[2]), 'visible_prob did not take'
assert n == int(sys.argv[3]), 'test_size did not take'
assert float(c.lidar.sensor_range) == 25.0, 'range must be DEFAULT'
# the two systems must carry their OWN prediction setting -- do not normalise
assert (pol == 'ours' and pm == 'truth') or (pol == 'height' and pm == 'none'), \
    'predict_method mismatch for %s: got %s' % (pol, pm)
PYEOF
  if [ $? -ne 0 ]; then echo "SKIP ${TAG}: verification failed"; return 1; fi

  python3 -u test.py --model_dir "$DST" --test_model "$CK" 2>&1 | tee "logs/re_${TAG}.log"
  rclone copy logs/ gdrive:HEIGHT_Backup/logs/ 2>&1 | grep -vE "NOTICE|Transferred|Checks|Elapsed" || true
}

cell 0.2 ours   data/exp_nospeak_s1 59522.pt
cell 0.2 height data/exp_heightpub  237800.pt
cell 1.0 ours   data/exp_nospeak_s1 59522.pt
cell 1.0 height data/exp_heightpub  237800.pt

step "RECIPROCITY ENDPOINTS at n=$N"
printf '  %-9s %-9s %s\n' "vis_prob" "policy" "successes / n"
for VP in 0.2 1.0; do for P in ours height; do
  T=${VP/./}
  R=$(grep -h "nav time (successes)" "logs/re_vp${T}_${P}.log" 2>/dev/null | tail -1 | sed 's/.*over //')
  printf '  %-9s %-9s %s\n' "$VP" "$P" "${R:-(missing)}"
done; done
echo
echo "  n=1000 sweep gave: ours 893->991, HEIGHT 868->987 (per 1000)"
echo "  PRE-REGISTERED: +9.8 / +11.9 pp rises; convergence at 1.0 (+0.4 pp, n.s.)"
rclone copy logs/ gdrive:HEIGHT_Backup/logs/ 2>&1 | grep -vE "NOTICE|Transferred|Checks|Elapsed" || true
