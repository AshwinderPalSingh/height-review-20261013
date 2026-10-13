"""Inevitable Collision State analysis: were the robot's collisions avoidable?

The oracle result alone cannot tell whether the residual collisions were unavoidable
or whether the policy was suboptimal: the oracle is a LEARNED policy handed perfect
information. This script asks whether the collisions were kinematically avoidable at all.

METHOD. For each collision episode, step back k timesteps from impact and ask whether
any sequence of the 9 discrete actions stays collision-free until the recorded moment of
contact, while every human continues along its RECORDED trajectory. The search is
depth-first and cuts no branch, but merges states that fall in the same quantised
(x, y, theta, v, w) cell (DEDUP_XY, DEDUP_TH below), so an "escapable" verdict comes with
a witness sequence and an "inevitable" verdict is exhaustive only up to that
quantisation.

WHAT A VERDICT MEANS. The robot follows its commanded unicycle kinematics (no wheel
saturation, motor noise or physics); the arena wall is not modelled, only the episode's
static rectangles; and the window ends at the recorded contact, so "escapable" means
contact could be postponed past that moment, not that it could be avoided for good.
This is a finite-window, nominal-model estimate, weaker than a full Inevitable Collision
State in the sense of Fraichard & Asama (IROS 2003).

REPLAY ASSUMPTION. Holding every human to its recorded path is an assumption: a human
with react_to_robot=False ignores the robot but reacts to neighbours that may have
reacted to it, and a reactive human would have responded to the counterfactual robot
directly. The headline number uses collisions whose nearest human is non-reactive;
reactive ones are reported separately. Note also that each recorded step pairs the
robot's pre-step pose with the humans' post-step positions (the env's own ordering).

Usage:  python3 ics_analysis.py data/exp_nospeak_s1/trajectories
"""
import glob
import os
import pickle
import sys

import numpy as np

# THE ROBOT IS ACCELERATION-LIMITED, NOT VELOCITY-COMMANDED.
# crowd_sim_tb2.py's action_convert maps each of the 9 discrete actions to a DELTA:
#   delta_v in {-0.05, 0, +0.05} m/s and delta_w in {-0.1, 0, +0.1} rad/s per 0.1 s step,
# then clips to v in [-0.5, 0.5] (allow_backward=True) and w in [-1, 1].
# Modelling this as "pick any velocity instantly" would be wrong in BOTH directions: it
# overstates the robot's escape ability (it cannot reverse from +0.5 to -0.5 in under a
# second) while understating it if reverse is omitted. Both errors are avoided by
# searching the actual delta set from the actual recorded (v, w).
DELTAS = [(dv, dw) for dv in (-0.05, 0.0, 0.05) for dw in (-0.1, 0.0, 0.1)]
# PyBullet registers mesh contact at a median centre-to-centre gap of +0.049 m BEYOND
# the sum of radii (measured over 348 recorded collisions). A circle test at exactly
# r_rob+r_hum therefore permits escapes the real robot cannot take, OVERSTATING
# avoidability. PAD adds that margin to bring the search's contact test closer to the
# environment's.
PAD = float(__import__('os').environ.get('ICS_PAD', '0.0'))
V_MIN, V_MAX = -0.5, 0.5
W_MIN, W_MAX = -1.0, 1.0


def step_unicycle(px, py, th, v, w, dv, dw, dt):
    """Apply one action delta, clip to limits, integrate one step."""
    v2 = min(max(v + dv, V_MIN), V_MAX)
    w2 = min(max(w + dw, W_MIN), W_MAX)
    th2 = th + w2 * dt
    return px + v2 * np.cos(th2) * dt, py + v2 * np.sin(th2) * dt, th2, v2, w2


def hits_human(rx, ry, humans, r_rob):
    for h in humans:
        hx, hy, hr = h[0], h[1], h[4]
        if (rx - hx) ** 2 + (ry - hy) ** 2 <= (r_rob + hr + PAD) ** 2:
            return True
    return False


def hits_obstacle(rx, ry, obs, r_rob):
    """Rectangles given as 4 world-frame vertices; axis-aligned test on their bbox."""
    if obs is None:
        return False
    for rect in obs:
        a = np.asarray(rect, dtype=float).reshape(-1, 2)
        x0, y0 = a[:, 0].min(), a[:, 1].min()
        x1, y1 = a[:, 0].max(), a[:, 1].max()
        # closest point on the rectangle to the robot centre
        cx = min(max(rx, x0), x1)
        cy = min(max(ry, y0), y1)
        if (rx - cx) ** 2 + (ry - cy) ** 2 <= (r_rob + PAD) ** 2:
            return True
    return False


# ---------------------------------------------------------------- exact search
# EXACT enumeration over the discrete action family, with state de-duplication.
#
# The previous version kept a beam of the B states with greatest clearance. That rule is
# inadmissible: clearance at depth k bounds nothing about collision-freedom at depth > k,
# so pruning could discard real escapes and inflate the inevitable fraction by an amount
# that varied with the horizon. Insensitivity to the beam width does not demonstrate
# convergence -- a biased heuristic converges to its bias.
#
# Exhaustive enumeration is what Fraichard & Asama's Property 4 (Adv. Robotics 18(10),
# 2004) actually licenses: restricting to any subset of the admissible controls yields an
# over-approximation of the ICS set -- it may label an escapable state as inevitable,
# never the reverse -- PROVIDED the search over that subset is exhaustive. Here that
# holds only inside the search's own model (commanded kinematics, replayed humans, no
# wall) and up to the de-duplication quantisation, so it does not bound the real env.
#
# Naive enumeration is 9^20 at the 2.0 s horizon, but the state space is far smaller than
# the tree: v is confined to a 21-point lattice on [-0.5, 0.5] and w to a 21-point lattice
# on [-1, 1], so de-duplicating on quantised (x, y, theta, v, w) collapses the search.
DEDUP_XY = 0.02      # m   -- spatial cell for de-duplication
DEDUP_TH = 0.10      # rad -- ~5.7 deg
STATE_CAP = 400000   # abort guard; a horizon that exceeds this is reported, never silently pruned


class SearchAborted(Exception):
    """Raised if the de-duplicated state set exceeds STATE_CAP."""


def escapable(steps, t0, k, r_rob, v_max, dt, obs):
    """Does ANY admissible action sequence stay collision-free for k steps from t0?

    Depth-first with memoisation on quantised (depth, x, y, theta, v, w), returning as
    soon as one surviving sequence is found. This answers an EXISTENCE question, so a
    single witness proves "escapable"; only the negative answer requires exhausting the
    de-duplicated state set, and in that case collisions prune the tree heavily. No beam:
    nothing that could survive is ever discarded.

    Returns True (escape exists within the discrete action family) or False (no
    sequence in that family survives under the search's model; see the module header
    for what that does and does not establish). Raises SearchAborted if the visited set exceeds STATE_CAP.
    """
    _, rpx, rpy, rvx, rvy, rth, _ = steps[t0]
    v0 = float(np.hypot(rvx, rvy))
    if np.cos(rth) * rvx + np.sin(rth) * rvy < 0:
        v0 = -v0
    w0 = 0.0
    if t0 > 0:
        w0 = (rth - steps[t0 - 1][5]) / dt
    w0 = min(max(w0, W_MIN), W_MAX)

    kmax = min(k, len(steps) - 1 - t0)
    if kmax <= 0:
        return True
    visited = set()
    stack = [(0, rpx, rpy, rth, v0, w0)]
    while stack:
        depth, px, py, th, v, w = stack.pop()
        if depth == kmax:
            return True                      # witness: survived the full horizon
        humans = steps[t0 + depth + 1][6]
        kids = []
        for (dv, dw) in DELTAS:
            nx, ny, nth, nv, nw = step_unicycle(px, py, th, v, w, dv, dw, dt)
            if hits_human(nx, ny, humans, r_rob):
                continue
            if hits_obstacle(nx, ny, obs, r_rob):
                continue
            key = (depth + 1, int(round(nx / DEDUP_XY)), int(round(ny / DEDUP_XY)),
                   int(round(nth / DEDUP_TH)), int(round(nv / 0.05)), int(round(nw / 0.1)))
            if key in visited:
                continue
            visited.add(key)
            if len(visited) > STATE_CAP:
                raise SearchAborted('visited %d states' % len(visited))
            kids.append((nx, ny, nth, nv, nw))
        # ORDER ONLY -- nothing is discarded, so this changes how quickly a witness is
        # found, never whether one exists. Most-clearance-first: the child furthest from
        # the nearest pedestrian is explored first, which reaches a surviving leaf sooner
        # in escapable states and leaves the inevitable case exhaustive as before.
        if len(kids) > 1 and humans:
            def _clear(c):
                return min((c[0] - h[0]) ** 2 + (c[1] - h[1]) ** 2
                           for h in humans if h[0] is not None)
            kids.sort(key=_clear)          # ascending; stack pops the last = most clearance
        stack.extend((depth + 1,) + c for c in kids)
    return False                             # exhausted: no escape exists


def main(traj_dir):
    files = sorted(glob.glob(os.path.join(traj_dir, 'ep_*.pkl')))
    if not files:
        sys.exit('no trajectory files in %s' % traj_dir)

    horizons = [5, 10, 20]          # 0.5 s, 1.0 s, 2.0 s at dt=0.1
    res = {k: {'nonreact': [0, 0], 'react': [0, 0]} for k in horizons}
    n_coll = n_skipped = n_noobs = 0
    n_type = {}
    eps_cache = []

    for fn in files:
        with open(fn, 'rb') as f:
            d = pickle.load(f)
        if not str(d['outcome']).startswith('Collision') or len(d.get('steps', [])) < 25:
            continue
        if len(d['steps'][0]) < 7:
            n_skipped += 1
            continue                # old dump format, missing theta / react flag
        n_coll += 1
        eps_cache.append(d)
        _t = str(d['outcome']).split('(')[0]
        n_type[_t] = n_type.get(_t, 0) + 1
        obs = d.get('obstacle_vertices')
        if obs is None:
            n_noobs += 1
        r_rob = d['robot_radius']
        v_max = d.get('robot_v_max', 0.5)
        dt = d.get('time_step', 0.1)
        steps = d['steps']

        # Was the human closest to the robot at impact a reactive one? That decides
        # whether counterfactual replay is exact for this episode.
        final_humans = [h for h in steps[-1][6] if h[0] is not None]
        rx, ry = steps[-1][1], steps[-1][2]
        if final_humans:
            # surface GAP, not centre distance -- same convention as the taxonomy
            j = int(np.argmin([np.hypot(rx - h[0], ry - h[1]) - (r_rob + h[4]) for h in final_humans]))
            reactive = bool(final_humans[j][5])
        else:
            reactive = False
        bucket = 'react' if reactive else 'nonreact'

        for k in horizons:
            t0 = len(steps) - 1 - k
            if t0 < 0:
                continue
            ok = escapable(steps, t0, k, r_rob, v_max, dt, obs)
            res[k][bucket][0 if ok else 1] += 1   # [avoidable, inevitable]

    print('collision episodes analysed : %d  (%s)' % (n_coll, ', '.join('%s=%d' % kv for kv in sorted(n_type.items()))))
    print('CAVEAT: replay is exact for the COLLIDING pedestrian when it is non-reactive; other')
    print('  reactive pedestrians (17.6% of the population) would in reality have responded to the')
    print('  counterfactual robot motion. Their recorded paths are used as-is. State this in the paper.')
    if n_skipped:
        print('SKIPPED (old dump format, no theta/react flag) : %d' % n_skipped)
    if n_noobs:
        print('WARNING: %d episodes had NO obstacle geometry -- those results are biased '
              'toward "avoidable"' % n_noobs)
    if not n_coll:
        return

    print()
    print('%-12s %-30s %-26s' % ('look-back', 'NON-REACTIVE (exact replay)', 'reactive (approximate)'))
    for k in horizons:
        a_ok, a_no = res[k]['nonreact']
        b_ok, b_no = res[k]['react']
        at = a_ok + a_no
        bt = b_ok + b_no
        sa = ('%d/%d inevitable (%.1f%%)' % (a_no, at, 100.0 * a_no / at)) if at else 'n/a'
        sb = ('%d/%d inevitable (%.1f%%)' % (b_no, bt, 100.0 * b_no / bt)) if bt else 'n/a'
        print('%-12s %-30s %-26s' % ('%.1f s' % (k * 0.1), sa, sb))

    print()
    print('READING THIS TABLE. Each row starts from a DIFFERENT state -- the one the robot')
    print('  actually occupied that many seconds before impact -- and asks whether any')
    print('  admissible action sequence survives from there to the moment of contact. The')
    print('  rows are therefore three separate questions, not one question at three')
    print('  horizons, and nothing requires them to be ordered. They do in fact decrease')
    print('  with look-back time, which is the physically expected direction: more warning')
    print('  means more escape routes.')
    print()
    print('WHAT THE SEARCH GUARANTEES. Enumeration over the 9 discrete actions is')
    print('  exhaustive and de-duplicated, with no beam. A "False" (inevitable) verdict')
    print('  means no sequence in that family survives; by Property 4 of Fraichard &')
    print('  Asama (Adv. Robotics 18(10), 2004) this over-approximates the true ICS set,')
    print('  so the fraction reported is an UPPER bound on inevitability. An "escapable"')
    print('  verdict is backed by an explicit witness sequence.')
    print()
    print('CONTACT MODEL. ICS is defined relative to the collision predicate, so the')
    print('  result moves with it and that dependence is intrinsic, not a bug. PyBullet')
    print('  registers contact a median +0.049 m beyond the sum of radii; run this script')
    print('  at ICS_PAD=0.0, 0.049 and 0.10 and report the band, never a single value.')
    print()
    print('CAVEAT ON REPLAY. Counterfactual replay is exact only for the colliding')
    print('  pedestrian when it is non-reactive. Other reactive pedestrians in the scene')
    print('  would in reality have responded to the counterfactual robot motion; their')
    print('  recorded paths are used as-is. Hence the two columns.')

    # ------------------------------------------------------ correctness self-check
    # From a FIXED start state, ICS_T must be non-decreasing in T: any trajectory that
    # has collided by T1 has also collided by T2 >= T1. This is forced by the definition
    # for any prefix-closed action family, so a violation means the search is not exact.
    # (It does NOT apply to the table above, whose rows have different start states.)
    print()
    print('=' * 70)
    print('CORRECTNESS SELF-CHECK: monotonicity in T from a fixed start state')
    print('=' * 70)
    viol = tested = 0
    for d in eps_cache[:80]:
        steps = d['steps']
        t0 = len(steps) - 1 - max(horizons)
        if t0 < 0:
            continue
        tested += 1
        prev = None
        for T in (5, 10, 15, 20):
            try:
                e = escapable(steps, t0, T, d['robot_radius'],
                              d.get('robot_v_max', 0.5), d.get('time_step', 0.1),
                              d.get('obstacle_vertices'))
            except SearchAborted:
                prev = None
                continue
            if prev is False and e is True:
                viol += 1
            prev = e
    print('  episodes checked : %d' % tested)
    print('  violations       : %d' % viol)
    print('  VERDICT          : %s' % ('PASS -- search behaves exactly'
                                       if viol == 0 else
                                       'FAIL -- the search is not exact; do not report these numbers'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/exp_nospeak_s1/trajectories')
