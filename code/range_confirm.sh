#!/bin/bash
# Settle the 3 m cell at n=3500. Four evals, ~6 h.
#
#   tmux new-session -d -s rangeconfirm -c /workspace/HEIGHT \
#     "bash -l /workspace/HEIGHT/Experiments/range_confirm.sh 2>&1 | tee /workspace/HEIGHT/logs/RANGECONFIRM.log; exec bash"
#
# WHY. The n=1000 sweep gave a +2.70 pp oracle gap at 3 m (p=0.052) against +0.50 pp at
# 25 m (p=0.724). Three reasons that is not yet a finding:
#   - FIVE tests were run; P(at least one p<0.05 by chance) = 22.6%. Bonferroni threshold
#     is 0.010, and 0.052 does not clear it.
#   - The curve is NOT monotone: 2.70 -> 0.80 -> 1.90 -> 1.60 -> 0.50. A physical range
#     effect does not zigzag; noise does.
#   - n=1000 is underpowered. The same 2.7 pp effect would give p=0.0003 at n=3500.
#
# So this is not "prove the null" -- it is "the largest cell is fragile, settle it".
# Only the two extreme ranges are re-run: 3 m (largest gap) and 25 m (reference, and the
# range every policy was actually trained at).
#
# ALL POLICIES REMAIN OUT OF DISTRIBUTION at 3 m (trained at 25 m). Only the WITHIN-RANGE
# oracle-minus-occluded gap is interpretable; absolute rates are not comparable across
# ranges or to any other result in this project.
#
# DECISION RULE, fixed in advance so the outcome is not read selectively:
#   3 m gap >= ~2 pp AND p < 0.01  -> range dependence is real; 21 Sep trains at 3 m
#   otherwise                      -> 21 Sep runs nopredict_350M.sh as planned
cd /workspace/HEIGHT
mkdir -p logs
step() { echo; echo "############ $(date -u +%H:%M) :: $* ############"; }

SRC=data/exp_nospeak_s1
CK=59522.pt
N=3500

cell() {
  local RANGE=$1 ORACLE=$2
  local TAG="rc${RANGE}_$( [ "$ORACLE" = "True" ] && echo oracle || echo occl )"
  local DST="data/rc_${TAG}"
  step "range=${RANGE} m  oracle=${ORACLE}  n=${N}"

  rm -rf "$DST"; mkdir -p "$DST/checkpoints" "$DST/configs"
  cp "${SRC}/checkpoints/${CK}" "$DST/checkpoints/" || { echo "SKIP ${TAG}: missing ckpt"; return 1; }
  cp "${SRC}/configs/config.py" "$DST/configs/config.py" || { echo "SKIP ${TAG}: missing config"; return 1; }

  python3 set_config_value.py "$DST/configs/config.py" lidar.sensor_range "$RANGE"
  python3 set_config_value.py "$DST/configs/config.py" env.test_size "$N"
  python3 set_config_value.py "$DST/configs/config.py" sim.oracle_visibility "$ORACLE"
  python3 set_config_value.py "$DST/configs/config.py" training.output_dir "'${DST}'"
  python3 set_config_value.py "$DST/configs/config.py" cbf.enabled False
  python3 set_config_value.py "$DST/configs/config.py" env.traj_dump False

  python3 - "$DST" "$RANGE" "$ORACLE" "$N" <<'PYEOF'
import importlib, sys
m = sys.argv[1].replace('/', '.') + '.configs.config'
c = getattr(importlib.import_module(m), 'Config')()
got_r, got_o, got_n = float(c.lidar.sensor_range), str(getattr(c.sim,'oracle_visibility',False)), int(c.env.test_size)
print('  verify: sensor_range=%s oracle=%s test_size=%d occl=%s cbf=%s'
      % (got_r, got_o, got_n, c.sim.occlusion_method,
         getattr(getattr(c,'cbf',None),'enabled',None)))
assert got_r == float(sys.argv[2]), 'sensor_range did not take'
assert got_o == sys.argv[3], 'oracle_visibility did not take'
assert got_n == int(sys.argv[4]), 'test_size did not take'
PYEOF
  if [ $? -ne 0 ]; then echo "SKIP ${TAG}: verification failed"; return 1; fi

  python3 -u test.py --model_dir "$DST" --test_model "$CK" 2>&1 | tee "logs/rc_${TAG}.log"
  rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P >/dev/null 2>&1 || true
}

cell 3  False
cell 3  True
cell 25 False
cell 25 True

step "RESULT (n=$N per cell)"
printf '%-8s %-12s %-12s %s\n' "range" "occluded" "oracle" "gap"
for R in 3 25; do
  O=$(grep -h "nav time (successes)" "logs/rc_rc${R}_occl.log"   2>/dev/null | tail -1 | sed 's/.*over n=//')
  A=$(grep -h "nav time (successes)" "logs/rc_rc${R}_oracle.log" 2>/dev/null | tail -1 | sed 's/.*over n=//')
  if [ -n "$O" ] && [ -n "$A" ]; then
    printf '%-8s %-12s %-12s %s\n' "${R} m" \
      "$(python3 -c "print(f'{$O/$N*100:.2f}%')")" \
      "$(python3 -c "print(f'{$A/$N*100:.2f}%')")" \
      "$(python3 -c "
import math
n=$N; a=$A; o=$O
p1,p2=a/n,o/n; p=(a+o)/(2*n); se=math.sqrt(p*(1-p)*2/n)
z=(p1-p2)/se; pv=math.erfc(abs(z)/math.sqrt(2))
sed=math.sqrt(p1*(1-p1)/n+p2*(1-p2)/n); d=(p1-p2)*100
print(f'{d:+.2f} pp  CI[{d-1.96*sed*100:+.2f},{d+1.96*sed*100:+.2f}]  p={pv:.4f}')")"
  else
    printf '%-8s %-12s %-12s %s\n' "${R} m" "${O:-(missing)}" "${A:-(missing)}" "—"
  fi
done
echo
echo "  n=1000 screening was: 3 m +2.70 pp (p=0.052), 25 m +0.50 pp (p=0.724)"
echo "  trained reference at 25 m, n=3500: +1.17 pp (p=0.115)"
echo
echo "DECISION RULE (fixed in advance):"
echo "  3 m gap >= ~2 pp AND p < 0.01  -> range dependence REAL; 21 Sep trains at 3 m"
echo "  otherwise                      -> 21 Sep runs Experiments/nopredict_350M.sh"
rclone copy logs/ gdrive:HEIGHT_Backup/logs/ -P || true
