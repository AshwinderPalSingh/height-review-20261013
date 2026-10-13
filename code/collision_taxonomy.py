"""Per-collision taxonomy from recorded trajectories.

WHY THIS EXISTS. The exhaustive ICS analysis finds that most collisions were avoidable
given the pedestrians' actual futures (94.8% still had an escape 2 s before impact), and
the CBF experiment finds that a shield working from the present state recovers none of
them. Neither says WHICH collisions those are, or what distinguishes them. This
classifies every recorded collision along axes that the benchmark's own authors describe
qualitatively but never measure.

The claim this is built to test, from HEIGHT's failure analysis: the robot "typically
collides when a human arrives at its goal and suddenly changes directions to a new goal."
Nobody has quantified how often that actually happens. Axis (D) below does.

AXES
  A. Approach geometry at impact -- head-on / crossing / from-behind, from the bearing of
     the colliding human in the robot's frame.
  B. Reactivity of the colliding human (react_to_robot). With visible_prob=0.2 most
     pedestrians ignore the robot entirely; if collisions were uniformly distributed we
     would expect ~80% non-reactive. A departure from that is informative.
  C. Robot speed at impact, relative to v_max. Tests whether the robot was already
     braking (slow) or drove in at pace.
  D. Time since the colliding human last changed heading materially. This is the direct
     test of the stated failure mode.

CAVEAT. The "colliding human" is identified as the closest human at the final recorded
step. The dump does not record which human the simulator registered the collision with,
so for a tight cluster this attribution can be wrong. Reported counts should be read as
approximate; the distributional claims are robust to a few misattributions.

Usage:  python3 collision_taxonomy.py <trajectory_dir>
"""
import glob
import math
import os
import pickle
import sys

import numpy as np

HEADING_CHANGE_RAD = 0.5      # ~29 deg: a material change of direction
MIN_SPEED_FOR_HEADING = 0.05  # below this, heading is numerically meaningless


def load(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)


def colliding_human(step, robot_radius):
    """Index of the human closest to the robot at this step, plus the gap."""
    _, rpx, rpy = step[0], step[1], step[2]
    humans = step[6]
    best, best_d = None, 1e9
    for j, h in enumerate(humans):
        if h[0] is None:
            continue
        d = math.hypot(rpx - h[0], rpy - h[1]) - (robot_radius + h[4])
        if d < best_d:
            best, best_d = j, d
    return best, best_d


def approach_class(step, j):
    """Bearing of human j relative to the robot's heading, at impact."""
    _, rpx, rpy, rvx, rvy, rth = step[0], step[1], step[2], step[3], step[4], step[5]
    h = step[6][j]
    # bearing to the human in the robot's frame
    ang = math.atan2(h[1] - rpy, h[0] - rpx) - rth
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    a = abs(ang)
    if a < math.pi / 4:
        sector = 'front'
    elif a < 3 * math.pi / 4:
        sector = 'side'
    else:
        sector = 'behind'

    # relative motion: closing head-on vs crossing
    hvx, hvy = h[2], h[3]
    if hvx is None:
        return sector, 'unknown'
    # angle between the human's velocity and the robot->human vector
    rel = math.atan2(h[1] - rpy, h[0] - rpx)
    hv_ang = math.atan2(hvy, hvx)
    d = abs(((hv_ang - rel + math.pi) % (2 * math.pi)) - math.pi)
    kind = 'oncoming' if d > 2.0 else ('crossing' if d > 1.0 else 'receding')
    return sector, kind


def steps_since_heading_change(steps, j, thresh=HEADING_CHANGE_RAD):
    """How many steps back did human j last change heading by more than `thresh`?

    Returns None if the human never changed heading materially within the record, or if
    its speed was too low for heading to be meaningful.
    """
    n = len(steps)
    last = None
    prev_ang = None
    for i in range(n):
        hs = steps[i][6]
        if j >= len(hs):
            return None
        hvx, hvy = hs[j][2], hs[j][3]
        if hvx is None:
            continue
        spd = math.hypot(hvx, hvy)
        if spd < MIN_SPEED_FOR_HEADING:
            prev_ang = None
            continue
        ang = math.atan2(hvy, hvx)
        if prev_ang is not None:
            d = abs(((ang - prev_ang + math.pi) % (2 * math.pi)) - math.pi)
            if d > thresh:
                last = i
        prev_ang = ang
    if last is None:
        return None
    return n - 1 - last


def main(traj_dir):
    files = sorted(glob.glob(os.path.join(traj_dir, 'ep_*.pkl')))
    if not files:
        sys.exit('no trajectory files in %s' % traj_dir)

    rows = []
    skipped = 0
    for fn in files:
        d = load(fn)
        if not d.get('steps') or len(d['steps'][0]) < 7:
            skipped += 1
            continue
        # PEDESTRIAN collisions only. Obstacle/wall contacts have no 'colliding human';
        # an earlier version pooled all 420 collisions and reported the NEAREST bystander's
        # flag for the 72 obstacle/wall contacts (audit 17 Sep 2026). Set
        # TAXONOMY_ALL=1 to reproduce that pooled table.
        _want = 'Collision' if os.environ.get('TAXONOMY_ALL') else 'CollisionHuman'
        if not str(d['outcome']).startswith(_want):
            continue
        steps = d['steps']
        last = steps[-1]
        j, gap = colliding_human(last, d['robot_radius'])
        if j is None:
            skipped += 1
            continue
        sector, kind = approach_class(last, j)
        reactive = bool(last[6][j][5]) if len(last[6][j]) > 5 else None
        rspeed = math.hypot(last[3], last[4]) if last[3] is not None else float('nan')
        since = steps_since_heading_change(steps, j)
        dt = d.get('time_step', 0.1)
        rows.append(dict(outcome=d['outcome'], sector=sector, kind=kind,
                         reactive=reactive, rspeed=rspeed,
                         since_s=(None if since is None else since * dt)))

    if not rows:
        sys.exit('no collision episodes with usable trajectories')

    n = len(rows)
    print('collision episodes classified : %d  (skipped %d)' % (n, skipped))
    print()

    def dist(key, label):
        print(label)
        vals = {}
        for r in rows:
            vals[r[key]] = vals.get(r[key], 0) + 1
        for k in sorted(vals, key=lambda x: -vals[x]):
            print('  %-12s %4d  %5.1f%%' % (str(k), vals[k], 100.0 * vals[k] / n))
        print()

    dist('outcome', 'A. outcome type')
    dist('sector', 'B. where the human was, in the robot frame, at impact')
    dist('kind', 'C. relative motion at impact')

    # reactivity -- the interesting comparison is against the 20% base rate
    react = [r for r in rows if r['reactive'] is True]
    print('D. reactivity of the colliding human')
    print('   reactive     %4d  %5.1f%%' % (len(react), 100.0 * len(react) / n))
    print('   non-reactive %4d  %5.1f%%' % (n - len(react), 100.0 * (n - len(react)) / n))
    _base = 0.2 * (sum(dd / (dd + st) for dd in range(5, 10) for st in range(0, 3)) / 15)
    _p = (n - len(react)) / n
    _z = (_p - (1 - _base)) / math.sqrt(_base * (1 - _base) / n)
    _h = 2 * math.asin(math.sqrt(_p)) - 2 * math.asin(math.sqrt(1 - _base))
    print('   base rate: %.2f%% of pedestrians reactive (0.2 x E[dyn/total]; static are never'
          ' reactive), i.e. %.2f%% non-reactive' % (100 * _base, 100 * (1 - _base)))
    print('   z against base rate = %.2f   p = %.1e   Cohen h = %.2f'
          % (_z, math.erfc(abs(_z) / math.sqrt(2)), _h))
    print()

    sp = np.array([r['rspeed'] for r in rows if not math.isnan(r['rspeed'])])
    if sp.size:
        print('E. robot speed at impact (v_max = 0.5 m/s)')
        print('   mean %.3f   median %.3f   frac above 0.4 m/s: %.1f%%'
              % (sp.mean(), np.median(sp), 100.0 * (sp > 0.4).mean()))
        print('   Low speeds would indicate the robot was already braking into the')
        print('   collision; high speeds that it drove in at pace.')
        print()

    # THE headline axis: how recently did the colliding human change direction?
    since = [r['since_s'] for r in rows if r['since_s'] is not None]
    never = n - len(since)
    print('F. time since the colliding human last changed heading by >%.0f deg'
          % math.degrees(HEADING_CHANGE_RAD))
    print('   (this is the direct test of the stated failure mode:')
    print('    "a human arrives at its goal and suddenly changes directions")')
    if since:
        s = np.array(since)
        for thr in (0.5, 1.0, 2.0, 3.0):
            print('   within %.1f s of impact : %4d  %5.1f%%'
                  % (thr, int((s <= thr).sum()), 100.0 * (s <= thr).mean()))
        print('   median %.2f s   mean %.2f s' % (np.median(s), s.mean()))
    print('   never changed heading in the record : %d  (%.1f%%)'
          % (never, 100.0 * never / n))
    print()
    print('READ: a high fraction in the "within 1-2 s" rows would support the stated')
    print('  failure mode and motivate multi-modal goal prediction. A low fraction would')
    print('  mean collisions are dominated by steady-heading approaches instead, which')
    print('  points away from goal-change anticipation as the missing ingredient.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else 'data/exp_nospeak_s1/trajectories')
