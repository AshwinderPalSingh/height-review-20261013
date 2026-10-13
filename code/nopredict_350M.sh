#!/bin/bash
# PHASE 1: the DEPLOYABLE system. predict_method='none', 350M steps, 3 seeds. ~12 days.
#
#   tmux new-session -d -s np350 -c /workspace/HEIGHT \
#     "bash -l /workspace/HEIGHT/Experiments/nopredict_350M.sh 2>&1 | tee /workspace/HEIGHT/logs/NP350.log; exec bash"
#
# WHY. Every result so far used sim.predict_method='truth', which forward-simulates each
# pedestrian's ACTUAL ORCA policy and goal to obtain their real future trajectory --
# privileged simulator state no deployed robot has. HEIGHT's released config uses
# predict_method='none'. So the comparison has been unmatched, in OUR favour, and we
# still only tie (88.08 +- 0.49% vs 88.03%). This run produces the deployable number.
#
# BUDGET. 350M from scratch in ONE schedule. AUDIT 17 SEP: `nospeak` actually received
# ~426M (Stage-2 = 76M from scratch + a relaunch that reset the counter and ran a full
# 300M schedule; then the 50M fine-tune), so this arm is NOT matched to nospeak (it has
# 0.82x). It IS matched to HEIGHT-350M (height_ft150.sh), which is the comparison that
# matters for the deployable claim. Note 350M is still 1.75x HEIGHT's released 199.8M
# (237800 updates x 30 steps x 28 procs) -- DISCLOSE that; it makes any null conservative,
# since we spend more compute and still do not win.
#
# WHAT THIS ARM IS. occlusion_method stays 'vae', so this is HEIGHT + PaS occlusion
# inference, WITHOUT intention prediction. That completes a clean ladder:
#   HEIGHT (their ckpt)                 88.03%   no prediction, no occlusion
#   this run (HEIGHT + PaS)                  ?   no prediction, occlusion      [DEPLOYABLE]
#   nospeak (HEIGHT + PaS + intention)  88.08%   ORACLE prediction, occlusion  [oracle row]
#
# EXPECTATION, stated up front: most likely 87-88.5%. Removing privileged information
# cannot help. The value here is a legitimate comparison, not a higher number. A result
# ABOVE 88.08% would be genuinely interesting -- it would mean the prediction features
# were actively harmful -- but do not plan on it.
cd /workspace/HEIGHT
mkdir -p logs
step() { echo; echo "############ $(date -u +%H:%M) :: $* ############"; }

wait_for_train() {
  local NAME=$1 WAITED=0
  while [ ! -f "logs/${NAME}_DONE.flag" ]; do
    sleep 900; WAITED=$((WAITED+900))
    if ! tmux has-session -t "$NAME" 2>/dev/null; then
      echo "$(date -u +%H:%M) $NAME: session gone with no DONE flag -- abandoning this seed"
      return 1
    fi
    # 350M at ~1050 FPS is ~93 h; 150 h is a generous ceiling before we stop waiting
    if [ $WAITED -gt 540000 ]; then
      echo "$(date -u +%H:%M) $NAME: exceeded 150 h -- abandoning so the queue continues"
      return 1
    fi
    if [ $((WAITED % 10800)) -lt 900 ]; then
      echo "  $(date -u +%H:%M) $NAME still training: $(grep -E '^Updates' logs/${NAME}_train.log 2>/dev/null | tail -1)"
    fi
  done
  echo "$(date -u +%H:%M) $NAME: training finished"
}

for SEED in 1 2 3; do
  # run_experiment.sh derives its directory and DONE-flag names as exp_<cond>_s<seed>,
  # so use that exact name here. An earlier draft invented exp_nopredict350_s<seed>,
  # which made the resume check test a directory that could never exist and made
  # wait_for_train watch the wrong flag.
  NAME="exp_nopredict_s${SEED}"
  step "SEED $SEED — train $NAME (350M steps, ~4 days)"

  # ---- BUDGET GUARD (added 19 Sep). A 50M pilot ran under this exact name and left a
  # DONE flag. Trusting the flag alone would evaluate the 50M checkpoint as the 350M arm.
  # So: if a directory exists, read ITS config's budget; archive anything that is not
  # 350M, flag and all, before deciding anything.
  if [ -d "data/${NAME}" ]; then
    EXISTING_STEPS=$(python3 - "data/${NAME}" <<'PYEOF'
import importlib, sys
try:
    c = getattr(importlib.import_module(sys.argv[1].replace('/', '.') + '.configs.config'), 'Config')()
    print(int(c.training.num_env_steps))
except Exception as e:
    print(-1)
PYEOF
)
    if [ "$EXISTING_STEPS" != "350000000" ]; then
      TAG="${NAME}_${EXISTING_STEPS}steps_archived_$(date -u +%Y%m%d)"
      echo "data/${NAME} exists with budget ${EXISTING_STEPS} != 350M -- archiving as ${TAG}"
      tmux has-session -t "$NAME" 2>/dev/null && { echo "  a live session holds it; not archiving a running job. ABORT."; exit 1; }
      mv "data/${NAME}" "data/${TAG}"
      for f in logs/${NAME}_DONE.flag logs/${NAME}_train.log logs/eval_${NAME}.log logs/intrusion_test_${NAME}.json; do
        [ -f "$f" ] && mv "$f" "${f/${NAME}/${TAG}}" && echo "  moved $f"
      done
    fi
  fi

  if [ -f "logs/${NAME}_DONE.flag" ]; then
    echo "logs/${NAME}_DONE.flag exists for a 350M run -- training already complete, skipping to eval"
  elif [ -d "data/${NAME}" ] && tmux has-session -t "$NAME" 2>/dev/null; then
    echo "data/${NAME} exists and session $NAME is ALIVE -- a 350M run is in progress; waiting for it"
  elif [ -d "data/${NAME}" ]; then
    echo "data/${NAME} exists (350M budget) but no DONE flag and no live session: it died."
    echo "  Not resuming automatically -- archive or inspect it. Skipping this seed."
    continue
  else
    # never run two seeds at once: 28 procs each on one GPU halves both and muddles FPS
    while tmux ls 2>/dev/null | grep -q "^exp_nopredict_s"; do
      echo "  $(date -u +%H:%M) another nopredict seed is training; waiting before launching seed $SEED"
      sleep 900
    done
    NOPREDICT_STEPS=350e6 bash Experiments/run_experiment.sh nopredict "$SEED" \
      || { echo "LAUNCH FAILED: seed $SEED"; continue; }

    # run_experiment.sh PRINTS the budget but does not assert it. If the
    # NOPREDICT_STEPS override failed to reach the sed, the run would silently train
    # 50M and we would find out four days later. Check the run's own snapshot before
    # committing to the wait.
    python3 - "data/${NAME}" <<'PYEOF'
import importlib, os, sys
# train.py copies the live config into <output_dir>/configs early in startup and
# run_experiment.sh sleeps 150 s before returning, so the snapshot is normally there.
# Fall back to the live config if it is not -- that is the file train.py imports anyway.
d = sys.argv[1]
mod = (d.replace('/', '.') + '.configs.config'
       if os.path.exists(os.path.join(d, 'configs', 'config.py'))
       else 'crowd_nav.configs.config')
print('  reading budget from: %s' % mod)
c = getattr(importlib.import_module(mod), 'Config')()
steps = int(c.training.num_env_steps)
print('  budget check: %.0fM steps, %d updates, predict=%s, occl=%s'
      % (steps/1e6, steps // c.ppo.num_steps // c.training.num_processes,
         c.sim.predict_method, c.sim.occlusion_method))
assert steps == 350_000_000, 'BUDGET IS %d, EXPECTED 350M -- the override did not take' % steps
assert c.sim.predict_method == 'none', 'prediction is still on'
PYEOF
    if [ $? -ne 0 ]; then
      echo "!! seed $SEED launched with the WRONG budget -- killing it"
      tmux kill-session -t "$NAME" 2>/dev/null
      continue
    fi
  fi

  if wait_for_train "$NAME"; then
    step "SEED $SEED — evaluate at n=3500"
    CK=$(ls -t data/${NAME}/checkpoints/*.pt 2>/dev/null | grep -v _checkpoint | head -1 | xargs -r basename)
    if [ -n "$CK" ]; then
      python3 set_config_value.py "data/${NAME}/configs/config.py" env.test_size 3500
      python3 set_config_value.py "data/${NAME}/configs/config.py" cbf.enabled False
      python3 set_config_value.py "data/${NAME}/configs/config.py" env.traj_dump False
      # prove the arm is what we think it is before spending 3 h on it
      python3 - "data/${NAME}" <<'PYEOF'
import importlib, sys
m = sys.argv[1].replace('/', '.') + '.configs.config'
c = getattr(importlib.import_module(m), 'Config')()
print('  verify: predict=%s occl=%s acoustic=%s test_size=%s steps=%s'
      % (c.sim.predict_method, c.sim.occlusion_method,
         getattr(getattr(c, 'acoustic', None), 'enabled', None),
         c.env.test_size, c.training.num_env_steps))
assert c.sim.predict_method == 'none', 'NOT the deployable arm -- prediction is still on'
assert int(c.training.num_env_steps) == 350_000_000, 'WRONG BUDGET %d -- refusing to evaluate' % int(c.training.num_env_steps)
PYEOF
      if [ $? -ne 0 ]; then echo "SKIP eval seed $SEED: verification failed"; continue; fi
      python3 -u test.py --model_dir "data/${NAME}" --test_model "$CK" 2>&1 \
        | tee "logs/eval_${NAME}.log"
      rclone copy "data/${NAME}/checkpoints/${CK}" "gdrive:HEIGHT_Backup/checkpoints/${NAME}/" -P || true
    else
      echo "no checkpoint found for $NAME"
    fi
  fi
  rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P || true
done

step "PHASE 1 RESULTS — deployable system, predict_method='none', n=3500"
# The analysis is a standalone script so it can be re-run by hand at any time, with no
# assistant present, and produces paste-ready LaTeX for every \pending block in the paper.
if [ -f analyze_nopredict.py ]; then
  python3 analyze_nopredict.py --logs logs --seeds 1 2 3 2>&1 | tee logs/NOPREDICT_REPORT.txt
else
  echo "!! analyze_nopredict.py not found -- raw counts only:"
  for S in 1 2 3; do
    printf '    seed %s  ' "$S"
    grep -h "nav time (successes)" "logs/eval_exp_nopredict_s${S}.log" 2>/dev/null | tail -1 | sed 's/.*over //' || echo "(missing)"
  done
fi
rclone copy logs/ gdrive:HEIGHT_Backup/logs/ 2>&1 | grep -vE "NOTICE|Transferred|Checks|Elapsed" || true
echo
echo "To re-run the analysis at any time:  cd /workspace/HEIGHT && python3 analyze_nopredict.py"
echo "Full report saved to logs/NOPREDICT_REPORT.txt and synced to Drive."
