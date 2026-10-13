#!/bin/bash
# Re-dump trajectories at n=3500 so the ICS analysis rests on ~400 collisions, not 105.
#
#   tmux new-session -d -s icsboost -c /workspace/HEIGHT \
#     "bash -l /workspace/HEIGHT/Experiments/ics_boost.sh 2>&1 | tee /workspace/HEIGHT/logs/ICSBOOST.log; exec bash"
#
# WHY. Both deep-research assessments named this the weakest load-bearing number in the
# paper. The 37.3% inevitability figure comes from 105 collisions; a Wilson interval at
# n=105 is roughly [28%, 47%], wide enough that "over a third are inevitable" is
# compatible with "under 30%". At n=3500 we expect ~400 collisions and a CI near +-4.5 pp.
#
# It also enlarges the collision taxonomy (currently 105 events, the basis of the
# 97.1% / z=3.88 / p=1.0e-04 non-reciprocity result) by the same factor.
#
# Pure re-dump: same frozen checkpoint, same config, only test_size and traj_dump change.
cd /workspace/HEIGHT
mkdir -p logs
DIR=data/exp_nospeak_s1
CK=59522.pt

echo "############ $(date -u +%H:%M) :: ICS re-dump at n=3500 ############"
python3 set_config_value.py "$DIR/configs/config.py" env.test_size 3500
python3 set_config_value.py "$DIR/configs/config.py" env.traj_dump True
python3 set_config_value.py "$DIR/configs/config.py" cbf.enabled False
python3 set_config_value.py "$DIR/configs/config.py" robot.visible_prob 0.2
python3 set_config_value.py "$DIR/configs/config.py" sim.oracle_visibility False
python3 set_config_value.py "$DIR/configs/config.py" lidar.sensor_range 25

python3 - "$DIR" <<'PYEOF'
import importlib, sys
m = sys.argv[1].replace('/', '.') + '.configs.config'
c = getattr(importlib.import_module(m), 'Config')()
print('  verify: test_size=%s traj_dump=%s visible_prob=%s oracle=%s cbf=%s kinematics=%s'
      % (c.env.test_size, c.env.traj_dump, c.robot.visible_prob,
         getattr(c.sim,'oracle_visibility',None),
         getattr(getattr(c,'cbf',None),'enabled',None), c.action_space.kinematics))
assert int(c.env.test_size) == 3500
assert c.env.traj_dump is True, 'traj_dump off -> nothing to analyse'
assert float(c.robot.visible_prob) == 0.2, 'must be the BENCHMARK DEFAULT, not a swept value'
assert c.action_space.kinematics == 'turtlebot', 'non-turtlebot -> react_to_robot never True'
assert float(c.lidar.sensor_range) == 25.0, 'sensor_range drifted from the 25 m default'
assert str(getattr(c.sim,'oracle_visibility',False)) == 'False', 'oracle must be OFF for the default-condition dump'
assert not getattr(getattr(c,'cbf',None),'enabled',False), 'CBF must be OFF (cbf_sweep was killed mid-run)'
assert c.sim.predict_method == 'truth', 'predict_method drifted; this checkpoint was trained with truth'
PYEOF
if [ $? -ne 0 ]; then echo "ABORT: verification failed"; exit 1; fi

# a stale dump directory would silently mix old 105-collision episodes with new ones
if [ -d "${DIR}/trajectories" ]; then
  mv "${DIR}/trajectories" "${DIR}/trajectories_n1000_$(date -u +%Y%m%d)"
  echo "  archived the previous dump rather than mixing samples"
fi

python3 -u test.py --model_dir "$DIR" --test_model "$CK" 2>&1 | tee logs/ics_redump_eval.log

echo; echo "############ $(date -u +%H:%M) :: analysis ############"
NDUMP=$(ls "${DIR}/trajectories"/ep_*.pkl 2>/dev/null | wc -l)
echo "  trajectory files dumped: $NDUMP"

# Headline = PAD 0.049 (matches PyBullet contact registration), beam 600. Also the
# unpadded figure and a wider beam, so the paper can report the sensitivity honestly.
ICS_PAD=0.049 ICS_BEAM=600  python3 ics_analysis.py "${DIR}/trajectories" 2>&1 | tee logs/ics_redump_pad049_b600.log
ICS_PAD=0.049 ICS_BEAM=3000 python3 ics_analysis.py "${DIR}/trajectories" 2>&1 | tee logs/ics_redump_pad049_b3000.log
ICS_PAD=0.0   ICS_BEAM=600  python3 ics_analysis.py "${DIR}/trajectories" 2>&1 | tee logs/ics_redump_pad000_b600.log
echo
python3 collision_taxonomy.py   "${DIR}/trajectories" 2>&1 | tee logs/ics_redump_taxonomy.log
python3 taxonomy_sensitivity.py "${DIR}/trajectories" 2>&1 | tee logs/ics_redump_taxsens.log
python3 responsibility.py       "${DIR}/trajectories" 2>&1 | tee logs/ics_redump_responsibility.log

# leave traj_dump OFF -- it costs disk and time on every later eval
python3 set_config_value.py "$DIR/configs/config.py" env.traj_dump False

rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P || true
# NEW folder: the old exp_nospeak_s1_n3500/ dump has the wrong-episode obstacles and
# must never be mixed with this one.
rclone copy "${DIR}/trajectories/" "gdrive:HEIGHT_Backup/trajectories/exp_nospeak_s1_n3500_fixedobs/" -P || true
echo
echo "READ (audit 17 Sep):"
echo "  ICS: there is NO valid prior figure -- every earlier ICS number used wrong-episode"
echo "       obstacles. Headline = pad049_b600. Report pad049_b3000 and pad000_b600 alongside."
echo "       Paste into paper3.tex Sec. VI with the three caveats listed in its pending block."
echo "  Taxonomy (pedestrian collisions only) on the OLD dump was: 347/348 = 99.7% non-reactive,"
echo "       z=8.49, h=0.76; bearing 62.1/23.3/14.7; motion 73.9/24.1/2.0; speed 0.255."
echo "       This is a NEW draw of the same policy, so expect small moves. Update Tables VI-VIII"
echo "       and Sec. X-B from THESE logs; do not mix the two dumps."
echo "  Responsibility on the OLD dump: R_robot 0.818, >=0.5 in 93.3%, first evasion 7.10 s,"
echo "       peak 2.20 s; control +0.061 (p=0.005)."
