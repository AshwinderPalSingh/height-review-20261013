#!/bin/bash
# DOSE-RESPONSE: navigation success vs pedestrian reciprocity.
#
#   tmux new-session -d -s recip -c /workspace/HEIGHT \
#     "bash -l /workspace/HEIGHT/Experiments/reciprocity_sweep.sh 2>&1 | tee /workspace/HEIGHT/logs/RECIP.log; exec bash"
#
# WHY. Two independent deep-research assessments both named this as the single experiment
# most likely to move the paper from "interesting but thin" to "accept". The collision
# taxonomy found 97.1% of collisions involve NON-REACTIVE pedestrians against an 82.9%
# base rate (z=3.88, p=1.0e-04) -- but that is a post-hoc taxonomy at ONE operating point.
# A monotone success-vs-reciprocity curve converts it into a dose-response relationship,
# which is categorically stronger causal evidence AND is immune to the nearest-pedestrian
# attribution concern, because it needs only aggregate rates, never per-event attribution.
#
# THE FALSIFIABLE TEST (state it in the paper BEFORE reporting the result):
#   If visible_prob=1.0 lifts success far above 88%, the non-reciprocity thesis holds.
#   If visible_prob=1.0 leaves success near 88%, THE THESIS IS FALSE and the binding
#   constraint is kinematics/density instead. Either outcome is publishable; pre-register
#   the prediction so the result cannot be read selectively.
#
# WHAT visible_prob ACTUALLY DOES (verified in crowd_sim_var_human.py:281-288):
#   if robot.kinematics == 'turtlebot':
#       if static: react_to_robot = False        # static humans NEVER react
#       else:      react_to_robot = (rand() <= visible_prob)
#   A reacting human gets the robot inserted into its ORCA observation
#   (crowd_sim.py:1246), so it avoids the robot. A non-reacting human does not see the
#   robot at all -- it is causally outside the interaction loop.
#
#   NOTE the static-human carve-out: population reactive share is
#   visible_prob * dynamic/(dynamic+static) = visible_prob * 6/7 at mean composition,
#   NOT visible_prob. Report the sweep against TRUE reactive share, not the raw parameter.
#
# EVAL-ONLY on frozen checkpoints. No training. Both policies are evaluated at every
# point so the curve is not a single-architecture artifact.
cd /workspace/HEIGHT
mkdir -p logs
step() { echo; echo "############ $(date -u +%H:%M) :: $* ############"; }

N=1000   # screening; the two endpoints get re-run at 3500 by reciprocity_confirm.sh

cell() {
  local VP=$1 WHO=$2 SRC=$3 CK=$4
  local TAG="vp${VP/./}_${WHO}"
  local DST="data/rp_${TAG}"
  step "visible_prob=${VP}  policy=${WHO}"

  rm -rf "$DST"; mkdir -p "$DST/checkpoints" "$DST/configs"
  cp "${SRC}/checkpoints/${CK}" "$DST/checkpoints/" || { echo "SKIP ${TAG}: no ckpt"; return 1; }
  cp "${SRC}/configs/config.py" "$DST/configs/config.py" || { echo "SKIP ${TAG}: no config"; return 1; }

  python3 set_config_value.py "$DST/configs/config.py" robot.visible_prob "$VP"
  python3 set_config_value.py "$DST/configs/config.py" env.test_size "$N"
  python3 set_config_value.py "$DST/configs/config.py" training.output_dir "'${DST}'"
  python3 set_config_value.py "$DST/configs/config.py" cbf.enabled False
  python3 set_config_value.py "$DST/configs/config.py" env.traj_dump False
  python3 set_config_value.py "$DST/configs/config.py" sim.oracle_visibility False

  python3 - "$DST" "$VP" "$N" <<'PYEOF'
import importlib, sys
m = sys.argv[1].replace('/', '.') + '.configs.config'
c = getattr(importlib.import_module(m), 'Config')()
vp, n = float(c.robot.visible_prob), int(c.env.test_size)
# the turtlebot gate is what makes visible_prob live at all -- assert it
kin = c.action_space.kinematics
print('  verify: visible_prob=%s test_size=%d kinematics=%s robot.visible=%s'
      % (vp, n, kin, c.robot.visible))
assert vp == float(sys.argv[2]), 'visible_prob did not take: %s' % vp
assert n == int(sys.argv[3]), 'test_size did not take'
assert kin == 'turtlebot', 'kinematics != turtlebot -> react_to_robot is NEVER set True'
assert c.robot.visible is True, 'robot.visible False -> no human can react regardless'
PYEOF
  if [ $? -ne 0 ]; then echo "SKIP ${TAG}: verification failed"; return 1; fi

  python3 -u test.py --model_dir "$DST" --test_model "$CK" 2>&1 | tee "logs/rp_${TAG}.log"
  rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P >/dev/null 2>&1 || true
}

for VP in 0.0 0.2 0.4 0.6 0.8 1.0; do
  cell "$VP" ours   data/exp_nospeak_s1 59522.pt
  cell "$VP" height data/exp_heightpub  237800.pt
done

# restore the benchmark default so nothing downstream inherits a swept value
for D in data/exp_nospeak_s1 data/exp_heightpub; do
  python3 set_config_value.py "$D/configs/config.py" robot.visible_prob 0.2
  python3 set_config_value.py "$D/configs/config.py" env.test_size 3500
done

step "DOSE-RESPONSE: success vs pedestrian reciprocity (n=$N)"
printf '%-8s %-14s %-14s %s\n' "vis_prob" "true_react%" "OURS" "HEIGHT"
for VP in 0.0 0.2 0.4 0.6 0.8 1.0; do
  T=${VP/./}
  O=$(grep -h "nav time (successes)" "logs/rp_vp${T}_ours.log"   2>/dev/null | tail -1 | sed 's/.*over n=//')
  H=$(grep -h "nav time (successes)" "logs/rp_vp${T}_height.log" 2>/dev/null | tail -1 | sed 's/.*over n=//')
  TR=$(python3 -c "print(f'{$VP*6/7*100:.1f}%')")
  printf '%-8s %-14s %-14s %s\n' "$VP" "$TR" \
    "$( [ -n "$O" ] && python3 -c "print(f'{$O/$N*100:.1f}%')" || echo '-' )" \
    "$( [ -n "$H" ] && python3 -c "print(f'{$H/$N*100:.1f}%')" || echo '-' )"
done
echo
echo "  benchmark default is visible_prob=0.2 (true reactive share 17.1%)"
echo "  taxonomy at that point: 97.1% of collisions involve non-reactive pedestrians"
echo
echo "PRE-REGISTERED READING:"
echo "  monotone RISE toward vp=1.0  -> non-reciprocity IS the binding constraint; the"
echo "     taxonomy becomes a dose-response result and is the paper's second pillar."
echo "  FLAT curve                   -> the thesis is FALSE. The ceiling is kinematic or"
echo "     density-driven, and the paper must be reframed around the ICS axis instead."
rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P || true
