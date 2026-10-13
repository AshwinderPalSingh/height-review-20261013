#!/bin/bash
# HEIGHT's released checkpoint evaluated in a PRISTINE clone of HEIGHT's own repo.
#
#   bash -l /workspace/HEIGHT/Experiments/height_pristine.sh 2>&1 | tee /workspace/HEIGHT/logs/PRISTINE.log
#
# WHY. Every HEIGHT number in the paper was produced by running their checkpoint inside
# OUR codebase, with our additions gated off. A referee can reasonably ask whether the
# gating is complete. Running the same checkpoint in an untouched clone of their repo and
# getting the same success rate closes that objection.
#
# n=500 is deliberate: this is a CONSISTENCY check, not a new measurement. Our n=3500
# figure is 88.03%; the n=500 pristine number should sit inside roughly +-3 pp of that.
#
# STANDALONE, not in the chain: cloning a repo and building its deps can fail for reasons
# unrelated to the science, and a failure here should not confuse the Tier 2 output.
# Run it by hand; if it fails, debug it by hand.
set -u
WORK=/workspace/height_pristine
CKPT=/workspace/HEIGHT/data/exp_heightpub/checkpoints/237800.pt
mkdir -p /workspace/HEIGHT/logs
echo "=== $(date -u '+%F %H:%M') pristine HEIGHT check ==="

if [ ! -f "$CKPT" ]; then echo "checkpoint not found at $CKPT"; exit 1; fi

if [ ! -d "$WORK/.git" ]; then
  git clone https://github.com/Shuijing725/CrowdNav_HEIGHT.git "$WORK" || { echo "clone failed"; exit 1; }
fi
cd "$WORK"
echo "  repo HEAD: $(git rev-parse --short HEAD)  $(git log -1 --format=%cd --date=short)"

# Their test.py expects data/<model_dir>/configs/config.py and checkpoints/<ckpt>.
# Stage the released checkpoint under their OWN default config, untouched except for
# the two values test.py needs: which checkpoint, and how many episodes.
DST=data/pristine_check
rm -rf "$DST"; mkdir -p "$DST/checkpoints" "$DST/configs"
cp "$CKPT" "$DST/checkpoints/"
cp crowd_nav/configs/config.py "$DST/configs/config.py"
cp crowd_nav/configs/__init__.py "$DST/configs/__init__.py" || touch "$DST/configs/__init__.py"
touch data/__init__.py "$DST/__init__.py"
python3 - "$DST/configs/config.py" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r"(?m)^(\s*env\.test_size\s*=\s*).*$", r"\g<1>500", s)
s = re.sub(r"(?m)^(\s*training\.output_dir\s*=\s*).*$", r"\g<1>'data/pristine_check'", s)
open(p, 'w').write(s)
PYEOF
echo "  staged config; test_size and output_dir set, everything else THEIR default"
grep -E "^\s*(env\.test_size|robot\.visible_prob|lidar\.sensor_range|sim\.predict_method|sim\.arena_size)\s*=" "$DST/configs/config.py" | sed 's/^/    /'

# Assert the two edits actually took, and that THEIR config imports cleanly. If either
# fails, stop now rather than after an hour.
python3 - "$DST" <<'PYEOF'
import importlib, sys
m = sys.argv[1].replace('/', '.') + '.configs.config'
c = getattr(importlib.import_module(m), 'Config')()
n = int(c.env.test_size)
print('  verify: test_size=%d visible_prob=%s range=%s predict=%s arena=%s'
      % (n, c.robot.visible_prob, c.lidar.sensor_range,
         getattr(c.sim, 'predict_method', '?'), getattr(c.sim, 'arena_size', '?')))
assert n == 500, 'test_size edit did not take (got %d)' % n
assert float(c.robot.visible_prob) == 0.2, 'their default visible_prob is not 0.2?!'
PYEOF
[ $? -ne 0 ] && { echo "ABORT: their config did not stage/import as expected"; exit 1; }

# Their test.py must accept the same arguments ours does (ours is a fork of theirs).
if ! python3 test.py --help 2>&1 | grep -q -- "--model_dir"; then
  echo "ABORT: their test.py does not accept --model_dir; inspect it by hand:"
  python3 test.py --help 2>&1 | head -20
  exit 1
fi
echo "  their test.py accepts --model_dir; proceeding"

python3 -u test.py --model_dir "$DST" --test_model 237800.pt 2>&1 | tee /workspace/HEIGHT/logs/height_pristine.log

echo
echo "=== RESULT ==="
grep -hE "has success rate|nav time \(successes\)" /workspace/HEIGHT/logs/height_pristine.log | sed 's/.*INFO: /  /'
python3 - <<'PYEOF'
import math, re
txt = open('/workspace/HEIGHT/logs/height_pristine.log').read()
m = re.search(r'nav time \(successes\):.*?over n=(\d+)', txt)
if not m:
    print('  !! no final summary in the log -- the run did not complete'); raise SystemExit(1)
k = int(m.group(1)); n = 500
p_pristine = k / n
p_ours = 3081 / 3500
gap = (p_pristine - p_ours) * 100
# two-proportion z, n=500 vs n=3500
pp = (k + 3081) / (n + 3500); se = math.sqrt(pp * (1 - pp) * (1 / n + 1 / 3500))
z = (p_pristine - p_ours) / se; pv = math.erfc(abs(z) / math.sqrt(2))
print('  pristine repo, n=500   : %d/%d = %.2f%%' % (k, n, 100 * p_pristine))
print('  our codebase, n=3500   : 3081/3500 = 88.03%%')
print('  gap                    : %+.2f pp   z=%.2f   p=%.3f' % (gap, z, pv))
if abs(gap) <= 3.0 and pv > 0.05:
    print('  >> PASS: consistent. Gating is complete; the objection is closed.')
    print('     Paper sentence: "HEIGHT\'s checkpoint reproduces at %.1f%% (n=500) in an')
    print('     unmodified clone of the authors\' repository, consistent with the 88.03%%')
    print('     measured in our codebase (p=%.2f)."' % pv)
else:
    print('  >> FAIL: inconsistent. Do NOT cite any HEIGHT number until this is understood.')
    print('     Possible causes: different repo commit, different dependency versions,')
    print('     a config default that differs between their HEAD and their release.')
PYEOF
rclone copy /workspace/HEIGHT/logs/ gdrive:HEIGHT_Backup/logs/ 2>&1 | grep -vE "NOTICE|Transferred|Checks|Elapsed" || true
