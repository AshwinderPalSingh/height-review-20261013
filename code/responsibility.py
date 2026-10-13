"""Responsibility metric for social robot navigation.

Adapts the formulation of Wenzel & Probst, "Spotting the Unfriendly Robot --
Towards better Metrics for Interactions" (ICRA 2025 Workshop on Advances in Social
Navigation: Planning, HRI and Beyond), and of arXiv:2509.12890. Unlike the cited metric,
the robot's share below is normalised by the robot's and the human's combined
conflict-reducing contributions only; the cited metric's time term is omitted.

    CP(t)  = max(0, 1 - pDCE(t) / (r1 + r2))          conflict potential
    C(t)   = CP(t) * N(t)                             time-normalized conflict
    CC_x(t) ~ C(t) - C_{x, no change}(t)              counterfactual contribution
    R_x    = (1/C_total) * integral of CC_x^-(t) dt   responsibility

pDCE is the predicted distance at closest encounter under constant-velocity
extrapolation. CC_x is a counterfactual: conflict as it is, minus conflict had agent x
not changed its velocity since the previous step. A NEGATIVE contribution means the
agent acted to reduce conflict (cooperative); responsibility accumulates the negative
part, so R_robot near 1 means the robot supplied almost all of the conflict reduction
and R_robot near 0 means the pedestrian did.

Why it is reported here: a raw collision rate does not show which agent changed course
to reduce a conflict. R is a geometric score of that. It credits any velocity change
that lowers predicted conflict, whatever caused it (a goal change, a neighbour's
response), and never reads the reactivity flags, so it is not a measure of blame. It
is computed from recorded states, in which the robot lags the humans by one step.

Usage:
    python3 responsibility.py data/exp_nospeak_s1/trajectories
"""
import glob
import os
import pickle
import sys

import numpy as np


def pdce_and_tce(p1, v1, p2, v2):
    """Predicted distance and time at closest encounter, constant velocity.

    Returns (pDCE, TCE). If the agents are separating, TCE is 0 and pDCE is the
    current distance -- the encounter is now.
    """
    dp = np.asarray(p1, float) - np.asarray(p2, float)
    dv = np.asarray(v1, float) - np.asarray(v2, float)
    denom = float(dv @ dv)
    if denom < 1e-9:
        return float(np.linalg.norm(dp)), 0.0
    tce = -float(dp @ dv) / denom
    if tce < 0.0:
        return float(np.linalg.norm(dp)), 0.0
    return float(np.linalg.norm(dp + tce * dv)), tce


def conflict(p1, v1, r1, p2, v2, r2, horizon=5.0):
    """Time-normalized conflict C(t) between two agents."""
    dce, tce = pdce_and_tce(p1, v1, p2, v2)
    cp = max(0.0, 1.0 - dce / (r1 + r2))
    if cp <= 0.0:
        return 0.0
    # Time normalization: a conflict far in the future counts less than an imminent
    # one. Linear decay to zero at `horizon` seconds, per the time-normalization
    # extension in arXiv:2509.12890.
    n = max(0.0, 1.0 - tce / horizon)
    return cp * n


def episode_responsibility(steps, robot_radius, horizon=5.0):
    """Per-episode responsibility shares.

    Returns (R_robot, R_humans, C_total). Shares are normalized to sum to 1 over the
    positive contributions, following the paper; when no conflict occurs at all,
    returns (nan, nan, 0.0).
    """
    cc_robot_neg = cc_human_neg = 0.0
    cc_pos_total = 0.0

    for i in range(1, len(steps)):
        # steps gained a 7th field (robot theta) when the ICS dump was added; unpack
        # positionally so both the 6- and 7-field formats load.
        _p = steps[i - 1]; _c = steps[i]
        t_prev, rpx_p, rpy_p, rvx_p, rvy_p = _p[0], _p[1], _p[2], _p[3], _p[4]
        humans_p = _p[-1]
        t_cur, rpx, rpy, rvx, rvy = _c[0], _c[1], _c[2], _c[3], _c[4]
        humans = _c[-1]
        dt = t_cur - t_prev
        if dt <= 0 or len(humans) != len(humans_p):
            continue

        if None in (rpx, rpy, rvx, rvy, rvx_p, rvy_p):
            continue
        rp, rv = (rpx, rpy), (rvx, rvy)
        rv_prev = (rvx_p, rvy_p)

        for h, h_prev in zip(humans, humans_p):
            hpx, hpy, hvx, hvy, hr = h[0], h[1], h[2], h[3], h[4]
            if None in (hpx, hpy, hvx, hvy, h_prev[2], h_prev[3]):
                continue
            hp, hv = (hpx, hpy), (hvx, hvy)
            hv_prev = (h_prev[2], h_prev[3])

            c_now = conflict(rp, rv, robot_radius, hp, hv, hr, horizon)
            # Counterfactuals: what the conflict would be had each agent NOT changed
            # its velocity since the previous step.
            c_robot_nc = conflict(rp, rv_prev, robot_radius, hp, hv, hr, horizon)
            c_human_nc = conflict(rp, rv, robot_radius, hp, hv_prev, hr, horizon)

            cc_r = (c_now - c_robot_nc) * dt
            cc_h = (c_now - c_human_nc) * dt

            # Negative part = acted to REDUCE conflict. Positive part = intensified it.
            cc_robot_neg += min(0.0, cc_r)
            cc_human_neg += min(0.0, cc_h)
            cc_pos_total += max(0.0, cc_r) + max(0.0, cc_h)

    total_neg = cc_robot_neg + cc_human_neg
    if total_neg == 0.0:
        return float('nan'), float('nan'), cc_pos_total
    return cc_robot_neg / total_neg, cc_human_neg / total_neg, cc_pos_total


def evasion_timing(steps, robot_radius, horizon=5.0):
    """When does the robot's evasive effort occur, relative to the episode end?

    Returns (t_first_effort, t_peak_effort, t_end, peak_conflict), all seconds.
    t_first_effort is the first time the robot's counterfactual contribution is
    meaningfully negative (i.e. it acted to reduce conflict). If the robot only starts
    evading in the last ~1s before a collision, the failure is one of TIMING, not of
    effort or perception -- it is already too late for any control authority to help.
    """
    first = peak = None
    peak_mag = 0.0
    peak_conf = 0.0
    for i in range(1, len(steps)):
        # steps gained a 7th field (robot theta) when the ICS dump was added; unpack
        # positionally so both the 6- and 7-field formats load.
        _p = steps[i - 1]; _c = steps[i]
        t_prev, rpx_p, rpy_p, rvx_p, rvy_p = _p[0], _p[1], _p[2], _p[3], _p[4]
        humans_p = _p[-1]
        t_cur, rpx, rpy, rvx, rvy = _c[0], _c[1], _c[2], _c[3], _c[4]
        humans = _c[-1]
        dt = t_cur - t_prev
        if dt <= 0 or len(humans) != len(humans_p):
            continue
        if None in (rpx, rpy, rvx, rvy, rvx_p, rvy_p):
            continue
        rp, rv, rv_prev = (rpx, rpy), (rvx, rvy), (rvx_p, rvy_p)
        step_cc = 0.0
        for h in humans:
            hpx, hpy, hvx, hvy, hr = h[0], h[1], h[2], h[3], h[4]
            if None in (hpx, hpy, hvx, hvy):
                continue
            hp, hv = (hpx, hpy), (hvx, hvy)
            c_now = conflict(rp, rv, robot_radius, hp, hv, hr, horizon)
            c_nc = conflict(rp, rv_prev, robot_radius, hp, hv, hr, horizon)
            step_cc += (c_now - c_nc) * dt
            peak_conf = max(peak_conf, c_now)
        if step_cc < -1e-6:
            if first is None:
                first = t_cur
            if -step_cc > peak_mag:
                peak_mag, peak = -step_cc, t_cur
    t_end = steps[-1][0] if steps else float('nan')
    return first, peak, t_end, peak_conf


def main(traj_dir):
    files = sorted(glob.glob(os.path.join(traj_dir, 'ep_*.pkl')))
    if not files:
        sys.exit('no trajectory files in %s' % traj_dir)

    # Load once. The dump holds EVERY collision plus every 8th success (the control
    # set), so the headline must be restricted to human collisions explicitly: an
    # earlier version of this script pooled all 805 files and reported R over a mix of
    # collisions and successes (audit 17 Sep 2026). Obstacle/wall collisions are also
    # excluded from the headline -- the metric measures robot-vs-pedestrian conflict,
    # and in those episodes the pedestrian was not what the robot hit.
    eps = []
    for fn in files:
        with open(fn, 'rb') as f:
            d = pickle.load(f)
        if not d['steps']:
            continue
        eps.append(d)
    rows = []
    for d in eps:
        r_rob, r_hum, c_tot = episode_responsibility(d['steps'], d['robot_radius'])
        rows.append((d['episode'], str(d['outcome']), r_rob, r_hum, c_tot))

    valid = [r for r in rows if not np.isnan(r[2])]
    n_hc = sum(1 for d in eps if str(d['outcome']).startswith('CollisionHuman'))
    print('episodes in dump: %d  (human collisions %d, other collisions %d, successes %d)'
          % (len(eps), n_hc,
             sum(1 for d in eps if str(d['outcome']).startswith('Collision')) - n_hc,
             sum(1 for d in eps if str(d['outcome']) == 'ReachGoal')))
    hc_valid = [r for r in valid if r[1].startswith('CollisionHuman')]
    print('HEADLINE SET = human collisions with measurable conflict: %d of %d'
          % (len(hc_valid), n_hc))
    if not hc_valid:
        return

    r_rob = np.array([r[2] for r in hc_valid])
    print('R_robot  mean %.3f  median %.3f  std %.3f'
          % (r_rob.mean(), np.median(r_rob), r_rob.std(ddof=1)))

    for lo, hi, label in [(0.0, 0.2, 'robot barely evaded  (pedestrian-caused)'),
                          (0.2, 0.5, 'robot partly evaded'),
                          (0.5, 1.01, 'robot did most of the evading')]:
        n = int(((r_rob >= lo) & (r_rob < hi)).sum())
        print('  R_robot in [%.1f,%.1f) : %5d  %5.1f%%   %s'
              % (lo, hi, n, 100.0 * n / len(hc_valid), label))

    # Timing: how long BEFORE the collision does the robot start / peak its evasion?
    # Human collisions only -- "before the collision" is meaningless for a success.
    hc_eps = [d for d in eps if str(d['outcome']).startswith('CollisionHuman')]
    lead_first, lead_peak = [], []
    for d in hc_eps:
        first, peak, t_end, _ = evasion_timing(d['steps'], d['robot_radius'])
        if first is not None:
            lead_first.append(t_end - first)
        if peak is not None:
            lead_peak.append(t_end - peak)
    if lead_first:
        lf, lp = np.array(lead_first), np.array(lead_peak)
        print('evasion timing (seconds BEFORE the collision), human collisions n=%d:' % lf.size)
        print('  first evasive action : median %.2f s  mean %.2f s' % (np.median(lf), lf.mean()))
        print('  peak evasive effort  : median %.2f s  mean %.2f s' % (np.median(lp), lp.mean()))
        print('  peak effort within 1.0s of impact : %.1f%%' % (100.0 * (lp < 1.0).mean()))
        print('  peak effort within 0.5s of impact : %.1f%%' % (100.0 * (lp < 0.5).mean()))

    # Horizon sensitivity. The time-normalization constant is NOT specified in the
    # accessible text of arXiv:2509.12890 -- 5.0s is our choice. The conclusion must be
    # shown stable across plausible horizons before it can be claimed.
    print('horizon sensitivity (R_robot, human collisions):')
    for hz in (3.0, 5.0, 7.5, 10.0):
        vals = []
        for d in hc_eps:
            rr, _, _ = episode_responsibility(d['steps'], d['robot_radius'], horizon=hz)
            if not np.isnan(rr):
                vals.append(rr)
        if vals:
            v = np.array(vals)
            print('  horizon %4.1f s : R_robot mean %.3f  median %.3f  n=%d  (>=0.5: %.1f%%)'
                  % (hz, v.mean(), np.median(v), v.size, 100.0 * (v >= 0.5).mean()))

    by_outcome = {}
    for _, out, rr, _, _ in valid:
        by_outcome.setdefault(out, []).append(rr)
    print('by outcome:')
    for out, vals in sorted(by_outcome.items()):
        print('  %-16s n=%4d  R_robot mean %.3f' % (out, len(vals), np.mean(vals)))

    # ---- THE CONTROL ----
    # With visible_prob=0.2 most pedestrians (about 82%) never respond to the robot, so a
    # high R_robot may reflect the environment design rather than a property of the
    # robot's behaviour in failures. If SUCCESS episodes show the same R_robot as
    # COLLISION episodes, the metric does not discriminate and no conclusion about the
    # failure mode can be drawn from it.
    succ = [rr for _, out, rr, _, _ in valid if out == 'ReachGoal']
    coll = [rr for _, out, rr, _, _ in valid if out.startswith('CollisionHuman')]
    print()
    print('=' * 68)
    print('CONTROL: does R_robot discriminate success from collision?')
    print('=' * 68)
    if not succ:
        print('  NO SUCCESS EPISODES DUMPED -- the control has not been run.')
        print('  Set config.env.traj_success_stride (default 8) and re-run test.py.')
        print('  Until then the failure-mode conclusion is CONFOUNDED, not supported.')
        return
    a, b = np.array(succ), np.array(coll)
    print('  success   : n=%4d  R_robot mean %.3f  sd %.3f' % (a.size, a.mean(), a.std(ddof=1)))
    print('  collision : n=%4d  R_robot mean %.3f  sd %.3f' % (b.size, b.mean(), b.std(ddof=1)))
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    t = diff / se if se > 0 else float('nan')
    # Welch-Satterthwaite df, normal approximation for p (df here is in the hundreds)
    import math
    pv = math.erfc(abs(t) / math.sqrt(2)) if se > 0 else float('nan')
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    print('  collision - success = %+.3f  95%% CI [%+.3f, %+.3f]  t=%.2f  p=%.4f'
          % (diff, lo, hi, t, pv))
    print()
    if not np.isfinite(pv):
        print('  => TEST DEGENERATE (zero variance in one group). Not interpretable.')
        return
    if pv < 0.05:
        print('  => R_robot DISCRIMINATES. The failure-mode reading is supported:')
        print('     collisions differ from successes in who does the conflict-resolution.')
        print('     Responsibility-weighted reward is justified as a next step.')
    else:
        print('  => R_robot does NOT discriminate (p=%.3f). It is measuring the 80%%' % pv)
        print('     non-reciprocity of the environment, NOT a property of failures.')
        print('     DROP the "ineffective evasion" claim. The ceiling + power critique')
        print('     stand on their own; the mechanism story does not.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/exp_nospeak_s1/trajectories')
