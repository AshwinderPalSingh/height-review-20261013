#!/bin/bash
# COMPUTE-MATCHED HEIGHT: fine-tune their released checkpoint by 150M steps to 350M total.
# AUDIT 17 SEP: 350M matches the nopredict arm (nopredict_350M.sh), NOT `nospeak`, which
# received ~426M (76M + 300M restart + 50M). Against nospeak this arm still has 0.82x.
#
#   tmux new-session -d -s heightft -c /workspace/HEIGHT \
#     "bash -l /workspace/HEIGHT/Experiments/height_ft150.sh 2>&1 | tee /workspace/HEIGHT/logs/HEIGHTFT.log; exec bash"
#
# WHY. Our system trained 350M steps; HEIGHT's released checkpoint is at 199.8M. The one
# positive result in the paper -- 16% fewer personal-space intrusions -- is confounded by
# that 150M gap: a longer-trained policy may simply be smoother. This closes the gap.
#
# DESIGN. Fine-tune, not retrain-from-scratch. Both systems then share the same recipe
# ("warm-started, fine-tuned to 350M"), which is the tighter comparison. From scratch
# would add an initialisation confound instead of removing one.
#
# READ THE RESULT AS:
#   HEIGHT-350M intrusion ratio stays ~5.66  -> the 16% is ARCHITECTURAL; confound dead
#   drops toward 4.75                        -> the 16% was COMPUTE; withdraw the claim
#   lands between                            -> partial; report the split
#
# TRAP AVOIDED. train.py loads <stem>_checkpoint.pt (optimizer + start_epoch) if it sits
# beside the weights. If HEIGHT's release has one, start_epoch=237800 would exceed a 150M
# num_updates and the loop would run ZERO times. The checkpoint is therefore copied alone
# into a clean staging dir so start_epoch is always 0.
#
# ~150M / ~1050 FPS ~ 40 h, then a 3 h eval. Finishes well before 21 Sep.
cd /workspace/HEIGHT
mkdir -p logs
step() { echo; echo "############ $(date -u '+%F %H:%M') :: $* ############"; }

NAME=exp_heightft
OUT=data/$NAME
SRC_CK=data/exp_heightpub/checkpoints/237800.pt
SRC_CFG=data/exp_heightpub/configs/config.py

if pgrep -f "python3 .*train.py" >/dev/null 2>&1; then echo "!! a training run is active -- refusing"; exit 1; fi
[ -f "$SRC_CK" ]  || { echo "missing $SRC_CK";  exit 1; }
[ -f "$SRC_CFG" ] || { echo "missing $SRC_CFG"; exit 1; }
[ -d "$OUT" ] && { echo "!! $OUT exists -- move it aside first: mv $OUT ${OUT}_old"; exit 1; }

step "stage checkpoint ALONE (no _checkpoint.pt sibling) and config"
mkdir -p "$OUT/init" "$OUT/configs" "$OUT/checkpoints"
cp "$SRC_CK" "$OUT/init/237800.pt"
ls "$OUT/init/"
[ -f "$OUT/init/237800_checkpoint.pt" ] && { echo "!! sibling present -- must not be"; exit 1; }
cp "$SRC_CFG" "$OUT/configs/config.py"
cp crowd_nav/configs/__init__.py "$OUT/configs/__init__.py" 2>/dev/null || touch "$OUT/configs/__init__.py"
touch data/__init__.py "$OUT/__init__.py"

# Also install as the LIVE config, since train.py imports crowd_nav.configs.config.
cp crowd_nav/configs/config.py crowd_nav/configs/config.py.bak_heightft
cp "$SRC_CFG" crowd_nav/configs/config.py
for C in crowd_nav/configs/config.py "$OUT/configs/config.py"; do
  python3 set_config_value.py "$C" training.resume "'rl'"
  python3 set_config_value.py "$C" training.load_path "'$OUT/init/237800.pt'"
  python3 set_config_value.py "$C" training.num_env_steps "150e6"
  python3 set_config_value.py "$C" training.output_dir "'$OUT'"
  python3 set_config_value.py "$C" sim.occlusion_method "'none'"
  python3 set_config_value.py "$C" sim.predict_method "'none'"
  python3 set_config_value.py "$C" cbf.enabled False
  python3 set_config_value.py "$C" env.traj_dump False
done

step "verify the live config is plain HEIGHT, warm-started, 150M"
python3 - <<'PYEOF'
import importlib
c = getattr(importlib.import_module('crowd_nav.configs.config'), 'Config')()
upd = int(c.training.num_env_steps) // c.ppo.num_steps // c.training.num_processes
print('  occl=%s predict=%s resume=%s' % (c.sim.occlusion_method, c.sim.predict_method, c.training.resume))
print('  load_path=%s' % c.training.load_path)
print('  steps=%.0fM -> %d updates (HEIGHT release was 237800 updates = 199.8M)' % (int(c.training.num_env_steps)/1e6, upd))
print('  total after fine-tune ~ %.0fM' % ((237800*c.ppo.num_steps*c.training.num_processes + int(c.training.num_env_steps))/1e6))
assert c.sim.occlusion_method == 'none' and c.sim.predict_method == 'none', 'not plain HEIGHT'
assert c.training.resume == 'rl', 'not warm-starting'
assert int(c.training.num_env_steps) == 150_000_000, 'budget wrong'
assert c.training.load_path.endswith('exp_heightft/init/237800.pt'), 'load_path wrong'
PYEOF
[ $? -ne 0 ] && { echo "ABORT: config verification failed"; cp crowd_nav/configs/config.py.bak_heightft crowd_nav/configs/config.py; exit 1; }

step "PRE-FLIGHT: does EVERY checkpoint key transfer into the model this config builds?"
# train.py does a FILTERED load -- any key mismatched in name or shape is silently
# dropped and left randomly initialised, yet it still prints "Loaded the following
# checkpoint". If the architectures differ at all, the fine-tune would be partly from
# scratch and nothing would tell us. Build the model exactly as train.py does and
# demand a 100% key match before spending 40 hours.
python3 - <<'PYEOF'
import importlib, torch, gym, crowd_sim
from training.networks.envs import make_vec_envs
from training.networks.model import Policy
c = getattr(importlib.import_module('crowd_nav.configs.config'), 'Config')()
envs = make_vec_envs('CrowdSim3DTbObs-v0', c.env.seed, 1, c.reward.gamma, None, 'cpu', False, config=c, ax=None)
model = Policy(envs.observation_space.spaces, envs.action_space, base_kwargs=c, base=c.robot.policy)
ck = torch.load(c.training.load_path, map_location='cpu')
md = model.state_dict()
matched = [k for k, v in ck.items() if k in md and v.size() == md[k].size()]
missing_in_model = [k for k in ck if k not in md]
shape_mismatch  = [k for k in ck if k in md and ck[k].size() != md[k].size()]
fresh_in_model  = [k for k in md if k not in ck]
print('  checkpoint keys      : %d' % len(ck))
print('  model keys           : %d' % len(md))
print('  transferred          : %d' % len(matched))
print('  in ckpt, not in model: %d %s' % (len(missing_in_model), missing_in_model[:4]))
print('  shape mismatch       : %d %s' % (len(shape_mismatch), [(k, tuple(ck[k].shape), tuple(md[k].shape)) for k in shape_mismatch[:3]]))
print('  in model, not in ckpt: %d %s' % (len(fresh_in_model), fresh_in_model[:4]))
ok = (len(matched) == len(ck) == len(md))
print('  => %s' % ('100%% MATCH -- safe to fine-tune' if ok else 'INCOMPLETE -- the fine-tune would be partly from scratch'))
raise SystemExit(0 if ok else 1)
PYEOF
[ $? -ne 0 ] && { echo "ABORT: architecture mismatch; see above"; cp crowd_nav/configs/config.py.bak_heightft crowd_nav/configs/config.py; exit 1; }

step "train (~40 h)"
python3 -u train.py 2>&1 | tee "logs/${NAME}_train.log"
echo "EXIT=$?" > "logs/${NAME}_DONE.flag"
cp crowd_nav/configs/config.py.bak_heightft crowd_nav/configs/config.py   # restore live config

# the model must actually have moved on from where it started
grep -m1 "Loaded the following checkpoint" "logs/${NAME}_train.log" || echo "!! checkpoint load line NOT found -- did it warm-start?"
grep -E "^Updates" "logs/${NAME}_train.log" | tail -1

step "evaluate at n=3500 (per-episode intrusion logging on)"
CK=$(ls -t "$OUT"/checkpoints/*.pt 2>/dev/null | grep -v _checkpoint | head -1 | xargs -r basename)
[ -n "$CK" ] || { echo "no checkpoint produced"; exit 1; }
python3 set_config_value.py "$OUT/configs/config.py" env.test_size 3500
python3 set_config_value.py "$OUT/configs/config.py" lidar.sensor_range 25
python3 set_config_value.py "$OUT/configs/config.py" robot.visible_prob 0.2
python3 set_config_value.py "$OUT/configs/config.py" sim.oracle_visibility False
python3 -u test.py --model_dir "$OUT" --test_model "$CK" 2>&1 | tee "logs/eval_${NAME}.log"

step "RESULT — HEIGHT at 350M vs HEIGHT at 199.8M vs ours at 350M"
grep -hE "has success rate|intrusion ratio \(per-episode\)|path length:" "logs/eval_${NAME}.log" | sed 's/.*INFO: /  HEIGHT-350M  /'
echo "  HEIGHT-199.8M (released) : 88.03% success, intrusion 5.658 +/- 7.096"
echo "  ours-350M (oracle)       : 88.63% success, intrusion 4.747 +/- 6.138"
echo
echo "  READ: intrusion stays ~5.66 -> architectural. Drops toward 4.75 -> compute."
echo "        Test with: Welch t on the two per-episode JSONs (logs/intrusion_test_*.json)."
rclone copy logs/ gdrive:HEIGHT_Backup/logs/ 2>&1 | grep -vE "NOTICE|Transferred|Checks|Elapsed" || true
rclone copy "$OUT/checkpoints/$CK" "gdrive:HEIGHT_Backup/checkpoints/$NAME/" 2>&1 | grep -vE "NOTICE|Transferred|Checks|Elapsed" || true
