#!/bin/bash
# Matched HEIGHT-vs-ours comparison on SOCIAL metrics, with per-episode spread.
#
#   tmux new-session -d -s social -c /workspace/HEIGHT \
#     "bash -l /workspace/HEIGHT/Experiments/social_metrics.sh 2>&1 | tee /workspace/HEIGHT/logs/SOCIAL.log; exec bash"
#
# WHY. Every comparison in this project so far used success rate, where the two systems
# tie (+0.05 pp, t=0.17 over three seeds). But the existing n=3500 logs already contain a
# metric where they DO differ:
#
#     intrusion ratio   HEIGHT 5.66   ours 4.96   -> 12.4% relative reduction
#     path length       HEIGHT 10.32  ours 10.39  -> essentially tied
#
# Intrusion ratio is the fraction of timesteps the robot violates a pedestrian's personal
# space (dmin < discomfort_dist). It is a PRIMARY metric in social navigation --
# Francis et al., "Principles and Guidelines for Evaluating Social Robot Navigation"
# (arXiv 2306.16740), argues success rate alone is inadequate for this task.
#
# The problem: evaluation.py logged only the MEAN. A 12% relative difference is
# uninterpretable without the per-episode spread. This run adds that (and dumps raw
# per-episode values to JSON) so the difference can carry a real significance test.
#
# PRE-REGISTERED: the claim under test is "equal task success, fewer intrusions".
# If the per-episode SD turns out large enough that 5.66 vs 4.96 is not significant,
# THAT IS THE RESULT and the claim does not go in the paper.
#
# NOTE ON min_dist: it logs as 0.00 for both systems. That is NOT a bug -- dmin comes
# from PyBullet getClosestPoints()[8], a SURFACE-to-surface distance, so it goes to zero
# when bodies nearly touch. Report it as a surface metric or omit it; do not describe it
# as centre-to-centre separation.
cd /workspace/HEIGHT
mkdir -p logs
step() { echo; echo "############ $(date -u +%H:%M) :: $* ############"; }
N=3500

cell() {
  local TAG=$1 DIR=$2 CK=$3
  step "$TAG  (n=$N)"
  python3 set_config_value.py "$DIR/configs/config.py" env.test_size "$N"
  python3 set_config_value.py "$DIR/configs/config.py" cbf.enabled False
  python3 set_config_value.py "$DIR/configs/config.py" env.traj_dump False
  python3 set_config_value.py "$DIR/configs/config.py" sim.oracle_visibility False
  python3 set_config_value.py "$DIR/configs/config.py" robot.visible_prob 0.2
  python3 set_config_value.py "$DIR/configs/config.py" lidar.sensor_range 25

  python3 - "$DIR" "$N" <<'PYEOF'
import importlib, sys
m = sys.argv[1].replace('/', '.') + '.configs.config'
c = getattr(importlib.import_module(m), 'Config')()
print('  verify: test_size=%s sensor_range=%s visible_prob=%s oracle=%s cbf=%s discomfort=%s'
      % (c.env.test_size, c.lidar.sensor_range, c.robot.visible_prob,
         getattr(c.sim,'oracle_visibility',None),
         getattr(getattr(c,'cbf',None),'enabled',None), c.reward.discomfort_dist))
assert int(c.env.test_size) == int(sys.argv[2])
assert float(c.lidar.sensor_range) == 25.0, 'range must be the DEFAULT for a matched comparison'
assert float(c.robot.visible_prob) == 0.2, 'reciprocity must be the DEFAULT'
PYEOF
  if [ $? -ne 0 ]; then echo "SKIP $TAG: verification failed"; return 1; fi

  python3 -u test.py --model_dir "$DIR" --test_model "$CK" 2>&1 | tee "logs/soc_${TAG}.log"
  rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P >/dev/null 2>&1 || true
}

cell height data/exp_heightpub  237800.pt
cell ours   data/exp_nospeak_s1 59522.pt

step "SOCIAL METRICS (n=$N, matched: 25 m range, visible_prob=0.2, no shield)"
for t in height ours; do
  echo "--- $t ---"
  grep -hE "has success rate|intrusion ratio \(per-episode\)|min dist during intrusions|path length:" \
    "logs/soc_${t}.log" 2>/dev/null | sed 's/.*INFO: /  /'
done
echo
echo "Per-episode JSON dumps (for the offline significance test):"
ls -la logs/intrusion_test_*.json 2>/dev/null || echo "  (none written)"
rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P || true
echo
echo "NEXT: run the Welch t-test / Mann-Whitney on the two JSON dumps offline."
echo "  Intrusion ratios are per-episode percentages, heavily zero-inflated"
echo "  (many episodes have no intrusion at all), so report BOTH a Welch t-test and a"
echo "  Mann-Whitney U -- the t-test alone is not safe on that distribution."
