"""Sensitivity checks for the collision taxonomy. Addresses two referee objections.

  1. ATTRIBUTION. The colliding pedestrian is inferred as the NEAREST human at the final
     recorded step; the dump does not record which human the simulator registered. In a
     dense arena a second human can be almost as close, and if that second human has the
     opposite reactivity flag the attribution flips. This measures how often that
     ambiguity actually arises, and re-derives the non-reactive share under the
     PESSIMISTIC assignment (ambiguous cases resolved AGAINST the finding).

  2. HEADING-CHANGE THRESHOLD. The "never changed heading" statistic used a 29 degree
     (0.5 rad) threshold, which is arbitrary. This re-derives it at 20, 29, and 45 degrees
     to show the conclusion does not depend on the choice.

Usage:  python3 taxonomy_sensitivity.py <trajectory_dir>
"""
import glob
import math
import os
import pickle
import sys

import numpy as np

# visible_prob=0.2 applies to DYNAMIC humans only; static ones are hard-coded non-reactive.
# dynamic ~ U{5..9}, static ~ U{0,1,2} (crowd_sim_var_human.py:152-160), so the expected
# reactive share is 0.2 * E[dyn/(dyn+static)] = 0.2 * 0.8810 = 0.1762. The earlier 6/7*0.2
# = 0.1714 assumed exactly one static human and understated it.
BASE_REACT = 0.2 * (sum(d / (d + st) for d in range(5, 10) for st in range(0, 3)) / 15)


def load(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)


def gaps_at_impact(step, robot_radius):
    """Sorted (gap, index, reactive) for every human at the final step."""
    rpx, rpy = step[1], step[2]
    out = []
    for j, h in enumerate(step[6]):
        if h[0] is None:
            continue
        gap = math.hypot(rpx - h[0], rpy - h[1]) - (robot_radius + h[4])
        react = bool(h[5]) if len(h) > 5 else False
        out.append((gap, j, react))
    out.sort()
    return out


def heading_change_never(steps, j, thresh):
    prev = None
    for st in steps:
        hs = st[6]
        if j >= len(hs) or hs[j][2] is None:
            continue
        vx, vy = hs[j][2], hs[j][3]
        if math.hypot(vx, vy) < 0.05:
            prev = None
            continue
        ang = math.atan2(vy, vx)
        if prev is not None:
            d = abs(((ang - prev + math.pi) % (2 * math.pi)) - math.pi)
            if d > thresh:
                return False
        prev = ang
    return True


def main(traj_dir):
    files = sorted(glob.glob(os.path.join(traj_dir, 'ep_*.pkl')))
    cols = []
    for fn in files:
        d = load(fn)
        if not d.get('steps') or len(d['steps'][0]) < 7:
            continue
        if not str(d['outcome']).startswith('CollisionHuman'):
            continue
        cols.append(d)
    n = len(cols)
    if not n:
        sys.exit('no human-collision episodes found')
    print('human-collision episodes: %d\n' % n)

    # ------------------------------------------------------------- 1. attribution
    print('=' * 70)
    print('1. ATTRIBUTION SENSITIVITY')
    print('=' * 70)
    for margin in (0.05, 0.10, 0.20):
        ambiguous = 0
        flip_risk = 0          # ambiguous AND the runner-up has a DIFFERENT flag
        nonreact_optimistic = 0
        nonreact_pessimistic = 0
        for d in cols:
            g = gaps_at_impact(d['steps'][-1], d['robot_radius'])
            if not g:
                continue
            nearest = g[0]
            runner = g[1] if len(g) > 1 else None
            amb = runner is not None and (runner[0] - nearest[0]) <= margin
            if amb:
                ambiguous += 1
                if runner[2] != nearest[2]:
                    flip_risk += 1
            # optimistic: take the nearest as-is
            if not nearest[2]:
                nonreact_optimistic += 1
            # pessimistic: if ambiguous and EITHER candidate is reactive, count it reactive
            if amb:
                if not nearest[2] and not runner[2]:
                    nonreact_pessimistic += 1
            else:
                if not nearest[2]:
                    nonreact_pessimistic += 1
        po = nonreact_optimistic / n
        pp = nonreact_pessimistic / n
        se = math.sqrt(BASE_REACT * (1 - BASE_REACT) / n)
        base = 1 - BASE_REACT
        z_o = (po - base) / se
        z_p = (pp - base) / se
        print('margin %.2f m:' % margin)
        print('   ambiguous (2nd human within margin)   : %3d / %d  (%.1f%%)'
              % (ambiguous, n, 100.0 * ambiguous / n))
        print('   ambiguous AND flags differ (flip risk): %3d / %d  (%.1f%%)'
              % (flip_risk, n, 100.0 * flip_risk / n))
        print('   non-reactive share, nearest-as-is     : %.1f%%  z=%.2f'
              % (100 * po, z_o))
        print('   non-reactive share, PESSIMISTIC       : %.1f%%  z=%.2f  p=%.1e'
              % (100 * pp, z_p, math.erfc(abs(z_p) / math.sqrt(2))))
        print()
    print('READ: the PESSIMISTIC row resolves every ambiguous case AGAINST the finding.')
    print('  If it stays far above the %.1f%% base rate with a large z, the attribution'
          % (100 * (1 - BASE_REACT)))
    print('  method cannot be what produced the result.')
    print()

    # ------------------------------------------------------------- 2. threshold
    print('=' * 70)
    print('2. HEADING-CHANGE THRESHOLD SENSITIVITY')
    print('=' * 70)
    print('%-10s %-22s' % ('threshold', 'never changed heading'))
    for deg in (15, 20, 29, 45, 60):
        th = math.radians(deg)
        never = 0
        for d in cols:
            g = gaps_at_impact(d['steps'][-1], d['robot_radius'])
            if not g:
                continue
            if heading_change_never(d['steps'], g[0][1], th):
                never += 1
        print('%-10s %3d / %d  (%.1f%%)' % ('%d deg' % deg, never, n, 100.0 * never / n))
    print()
    print('READ: if "never changed heading" stays a clear majority across 15-60 deg, the')
    print('  claim that sudden goal-changes are a MINORITY mechanism is threshold-robust.')


if __name__ == '__main__':
    main(sys.argv[1])
