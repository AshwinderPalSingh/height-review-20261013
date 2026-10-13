"""Inevitable-collision search, version 2 (13 Oct 2026).

ics_analysis.py (v1) is kept unchanged: it produced ics_exact_results.txt and
ics_full_results.json, and it is what reproduces those files. This version changes four
things and nothing else.

1. CLOSED PREFIX. The window is [t0, t0+k]. The recorded start state is tested with the
   same human and rectangle predicates before anything else, so a start that is already
   inside the contact envelope is reported as `initial_contact` and never as escapable.
   v1 tested only the samples t0+1 .. t0+k and returned True for a zero horizon.
2. WITNESSES. A `survives_horizon` verdict carries the action sequence that survived, and
   replay_witness() re-simulates it step by step, without de-duplication, against both
   predicates. Every reported witness is replayed.
3. STATUSES. initial_contact, survives_horizon, no_surviving_sequence, aborted. Aborted
   searches are counted and reported, never dropped from the denominator silently.
4. ELIGIBILITY IS A STATED RULE. The historical cohort keeps v1's rule: a collision record
   needs at least 25 recorded states (391 episodes in the primary dump, 385 of them with a
   non-reactive nearest pedestrian). `--eligibility per-horizon` admits every collision
   record with at least k+1 states; that is a different cohort and is reported separately.

Kinematics, action set, contact predicates, PAD, de-duplication and search order are v1's,
so a verdict can differ from v1 only where the start state is already in contact.

The model is the same nominal one as v1: commanded unicycle kinematics (no wheel saturation
or physics), recorded pedestrian paths replayed unchanged, the episode's static rectangles
but not the arena wall, and states merged on a 2 cm / 0.1 rad grid. A no-surviving-sequence
verdict is exhaustive only up to that quantisation, and an initially violated envelope is a
margin violation under the chosen PAD, not proof of physical contact at that sample.

Usage:
    python3 ics_analysis_v2.py <trajectory_dir> --pad 0.049 --out ics_v2_pad0.049.json
    python3 ics_analysis_v2.py <trajectory_dir> --pad 0.049 --eligibility per-horizon --out ...
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

VERSION = 'ics_analysis_v2 2026-10-13'

DELTAS = [(dv, dw) for dv in (-0.05, 0.0, 0.05) for dw in (-0.1, 0.0, 0.1)]
V_MIN, V_MAX = -0.5, 0.5
W_MIN, W_MAX = -1.0, 1.0
DEDUP_XY = 0.02
DEDUP_TH = 0.10
STATE_CAP = 400000
HORIZONS = (5, 10, 20)          # 0.5 s, 1.0 s, 2.0 s at dt = 0.1
HISTORICAL_MIN_STEPS = 25       # v1's eligibility rule


class SearchAborted(Exception):
    pass


def step_unicycle(px, py, th, v, w, dv, dw, dt):
    v2 = min(max(v + dv, V_MIN), V_MAX)
    w2 = min(max(w + dw, W_MIN), W_MAX)
    th2 = th + w2 * dt
    return px + v2 * np.cos(th2) * dt, py + v2 * np.sin(th2) * dt, th2, v2, w2


def human_contact(rx, ry, humans, r_rob, pad):
    """Index of the first human whose padded envelope contains the robot centre, else None."""
    for j, h in enumerate(humans):
        if h[0] is None:
            continue
        if (rx - h[0]) ** 2 + (ry - h[1]) ** 2 <= (r_rob + h[4] + pad) ** 2:
            return j
    return None


def obstacle_contact(rx, ry, obs, r_rob, pad):
    """Index of the first rectangle (bounding box of its vertices) within r_rob + pad, else None."""
    if obs is None:
        return None
    for i, rect in enumerate(obs):
        a = np.asarray(rect, dtype=float).reshape(-1, 2)
        x0, y0 = a[:, 0].min(), a[:, 1].min()
        x1, y1 = a[:, 0].max(), a[:, 1].max()
        cx = min(max(rx, x0), x1)
        cy = min(max(ry, y0), y1)
        if (rx - cx) ** 2 + (ry - cy) ** 2 <= (r_rob + pad) ** 2:
            return i
    return None


def start_state(steps, t0, dt):
    _, rpx, rpy, rvx, rvy, rth, _ = steps[t0]
    v0 = float(np.hypot(rvx, rvy))
    if np.cos(rth) * rvx + np.sin(rth) * rvy < 0:
        v0 = -v0
    w0 = 0.0
    if t0 > 0:
        w0 = (rth - steps[t0 - 1][5]) / dt
    w0 = min(max(w0, W_MIN), W_MAX)
    return rpx, rpy, rth, v0, w0


def _search_from(steps, t0, kmax, r_rob, dt, obs, pad, start):
    """v1's traversal, unchanged in order and de-duplication, with the action path kept.

    Returns (True, path) when some sequence survives kmax steps, (False, None) when the
    de-duplicated set is exhausted. Only positive-time samples are tested here.
    """
    rpx, rpy, rth, v0, w0 = start
    if kmax <= 0:
        return True, ()
    visited = set()
    stack = [(0, rpx, rpy, rth, v0, w0, ())]
    while stack:
        depth, px, py, th, v, w, path = stack.pop()
        if depth == kmax:
            return True, path
        humans = steps[t0 + depth + 1][6]
        kids = []
        for a, (dv, dw) in enumerate(DELTAS):
            nx, ny, nth, nv, nw = step_unicycle(px, py, th, v, w, dv, dw, dt)
            if human_contact(nx, ny, humans, r_rob, pad) is not None:
                continue
            if obstacle_contact(nx, ny, obs, r_rob, pad) is not None:
                continue
            key = (depth + 1, int(round(nx / DEDUP_XY)), int(round(ny / DEDUP_XY)),
                   int(round(nth / DEDUP_TH)), int(round(nv / 0.05)), int(round(nw / 0.1)))
            if key in visited:
                continue
            visited.add(key)
            if len(visited) > STATE_CAP:
                raise SearchAborted('visited %d states' % len(visited))
            kids.append((nx, ny, nth, nv, nw, path + (a,)))
        if len(kids) > 1 and humans:
            def _clear(c):
                return min((c[0] - h[0]) ** 2 + (c[1] - h[1]) ** 2
                           for h in humans if h[0] is not None)
            kids.sort(key=_clear)
        stack.extend((depth + 1,) + c for c in kids)
    return False, None


def search(steps, t0, k, r_rob, dt, obs, pad):
    """Closed-prefix search over [t0, t0+k]. Returns a dict describing the verdict.

    `open_start` is what the same search says when the start sample is not tested (v1's
    semantics); it is kept so that initial envelope violations stay visible.
    """
    start = start_state(steps, t0, dt)
    kmax = min(k, len(steps) - 1 - t0)
    rec = {'t0': t0, 'kmax': kmax, 'status': None, 'witness': None,
           'initial_contact': None, 'open_start': None}
    hj = human_contact(start[0], start[1], steps[t0][6], r_rob, pad)
    oi = obstacle_contact(start[0], start[1], obs, r_rob, pad)
    try:
        ok, path = _search_from(steps, t0, kmax, r_rob, dt, obs, pad, start)
        rec['open_start'] = 'survives_horizon' if ok else 'no_surviving_sequence'
    except SearchAborted:
        ok, path = None, None
        rec['open_start'] = 'aborted'
    if hj is not None or oi is not None:
        rec['initial_contact'] = {'human': hj, 'obstacle': oi}
        rec['status'] = 'initial_contact'
        return rec
    if ok is None:
        rec['status'] = 'aborted'
    elif ok:
        rec['status'] = 'survives_horizon'
        rec['witness'] = list(path)
    else:
        rec['status'] = 'no_surviving_sequence'
    return rec


def replay_witness(steps, t0, actions, r_rob, dt, obs, pad):
    """Re-simulate a witness without de-duplication; True if every sample clears both predicates."""
    px, py, th, v, w = start_state(steps, t0, dt)
    if human_contact(px, py, steps[t0][6], r_rob, pad) is not None:
        return False
    if obstacle_contact(px, py, obs, r_rob, pad) is not None:
        return False
    for i, a in enumerate(actions):
        dv, dw = DELTAS[a]
        px, py, th, v, w = step_unicycle(px, py, th, v, w, dv, dw, dt)
        if human_contact(px, py, steps[t0 + i + 1][6], r_rob, pad) is not None:
            return False
        if obstacle_contact(px, py, obs, r_rob, pad) is not None:
            return False
    return True


def nearest_is_reactive(steps, r_rob):
    final_humans = [h for h in steps[-1][6] if h[0] is not None]
    if not final_humans:
        return False
    rx, ry = steps[-1][1], steps[-1][2]
    j = int(np.argmin([np.hypot(rx - h[0], ry - h[1]) - (r_rob + h[4]) for h in final_humans]))
    return bool(final_humans[j][5])


def eligible(d, eligibility, k):
    if not str(d['outcome']).startswith('Collision'):
        return False
    n = len(d.get('steps', []))
    if eligibility == 'historical':
        return n >= HISTORICAL_MIN_STEPS
    return n >= k + 1


def sha256_dir(files):
    h = hashlib.sha256()
    for fn in files:
        h.update(os.path.basename(fn).encode())
        with open(fn, 'rb') as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('traj_dir')
    ap.add_argument('--pad', type=float, required=True)
    ap.add_argument('--eligibility', choices=('historical', 'per-horizon'), default='historical')
    ap.add_argument('--selfcheck', type=int, default=-1,
                    help='episodes for the fixed-start monotonicity check (-1 = all eligible)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.traj_dir, 'ep_*.pkl')))
    if not files:
        sys.exit('no trajectory files in %s' % a.traj_dir)
    t_start = time.time()
    episodes, excluded = [], []
    counts = {k: {b: {s: 0 for s in ('initial_contact', 'survives_horizon',
                                     'no_surviving_sequence', 'aborted')}
                  for b in ('nonreact', 'react')} for k in HORIZONS}
    witness_fail = 0
    for fn in files:
        with open(fn, 'rb') as f:
            d = pickle.load(f)
        if not str(d['outcome']).startswith('Collision'):
            continue
        steps = d.get('steps', [])
        if len(steps) == 0 or len(steps[0]) < 7:
            excluded.append({'episode': d.get('episode'), 'reason': 'old dump format'})
            continue
        if not any(eligible(d, a.eligibility, k) for k in HORIZONS):
            excluded.append({'episode': d.get('episode'), 'outcome': str(d['outcome']),
                             'n_steps': len(steps), 'reason': 'too few recorded states'})
            continue
        r_rob = d['robot_radius']
        dt = d.get('time_step', 0.1)
        obs = d.get('obstacle_vertices')
        bucket = 'react' if nearest_is_reactive(steps, r_rob) else 'nonreact'
        ep = {'episode': d.get('episode'), 'outcome': str(d['outcome']), 'n_steps': len(steps),
              'bucket': bucket, 'has_obstacles': obs is not None, 'horizons': {}}
        for k in HORIZONS:
            if not eligible(d, a.eligibility, k):
                continue
            t0 = len(steps) - 1 - k
            rec = search(steps, t0, k, r_rob, dt, obs, a.pad)
            if rec['status'] == 'survives_horizon':
                rec['witness_replay_ok'] = replay_witness(steps, t0, rec['witness'], r_rob, dt, obs, a.pad)
                witness_fail += 0 if rec['witness_replay_ok'] else 1
            counts[k][bucket][rec['status']] += 1
            ep['horizons'][str(k)] = rec
        episodes.append((ep, d))

    # Fixed-start monotonicity: from t0 = len-1-20, a surviving sequence of length T2 has a
    # surviving prefix of every length T1 < T2, so a later survival after an earlier
    # no-surviving-sequence verdict means the search is not exact for that start.
    sc = episodes if a.selfcheck < 0 else episodes[:a.selfcheck]
    viol = tested = 0
    for ep, d in sc:
        steps = d['steps']
        t0 = len(steps) - 1 - max(HORIZONS)
        if t0 < 0:
            continue
        tested += 1
        seen_fail = False
        for T in (5, 10, 15, 20):
            r = search(steps, t0, T, d['robot_radius'], d.get('time_step', 0.1),
                       d.get('obstacle_vertices'), a.pad)
            if r['status'] == 'aborted':
                continue
            if r['status'] == 'survives_horizon' and seen_fail:
                viol += 1
            if r['status'] in ('no_surviving_sequence', 'initial_contact'):
                seen_fail = True

    out = {
        'version': VERSION,
        'pad': a.pad,
        'eligibility': a.eligibility,
        'traj_dir_sha256': sha256_dir(files),
        'n_files': len(files),
        'n_episodes': len(episodes),
        'excluded': excluded,
        'counts': {str(k): v for k, v in counts.items()},
        'witness_replay_failures': witness_fail,
        'selfcheck': {'episodes': tested, 'violations': viol},
        'seconds': round(time.time() - t_start, 1),
        'episodes': [ep for ep, _ in episodes],
    }
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)

    print('%s  pad=%.3f  eligibility=%s' % (VERSION, a.pad, a.eligibility))
    print('collision episodes searched: %d   excluded: %d' % (len(episodes), len(excluded)))
    for e in excluded:
        print('  excluded: %s' % e)
    print('%-9s %-48s %s' % ('look-back', 'non-reactive nearest pedestrian', 'reactive'))
    for k in HORIZONS:
        row = []
        for b in ('nonreact', 'react'):
            c = counts[k][b]
            n = sum(c.values())
            fail = c['initial_contact'] + c['no_surviving_sequence']
            row.append('%d/%d no escape (%.1f%%) [initial %d, none %d, aborted %d]' % (
                fail, n, 100.0 * fail / n if n else 0.0, c['initial_contact'],
                c['no_surviving_sequence'], c['aborted']) if n else 'n/a')
        print('%-9s %-48s %s' % ('%.1f s' % (k * 0.1), row[0], row[1]))
    print('witness replays failed: %d' % witness_fail)
    print('fixed-start monotonicity: %d episodes, %d violations' % (tested, viol))
    print('A survives_horizon verdict means one nominal action sequence stays outside the padded')
    print('envelope at every sample from the start through the last recorded state, under the')
    print('model described in the module header. It is not a certified escape in the simulator.')
    print('written to %s' % a.out)


if __name__ == '__main__':
    main()
