"""Collision taxonomy, version 2 (13 Oct 2026).

collision_taxonomy.py (v1) is kept unchanged; it produced collision_taxonomy_n3500.log and
the v1 rows of the paper's taxonomy table. This version changes one axis and its name.

v1's "relative motion" classes compare the colliding pedestrian's own velocity with the
robot-to-pedestrian bearing. The robot's velocity is not subtracted, so the classes
describe the pedestrian's direction of travel relative to the robot, not relative motion:
a robot closing at 0.5 m/s on a pedestrian walking away at 0.2 m/s is "receding". The axis
is renamed "pedestrian approach direction" and keeps v1's thresholds (angle between the
pedestrian's velocity and the robot-to-pedestrian bearing above 2 rad: oncoming; above
1 rad: crossing; otherwise receding). A pedestrian with no usable velocity was classified
by atan2(0, 0) = 0, i.e. by the bearing alone; v2 classes it as stationary when its speed
is at most STATIONARY_SPEED (1e-8 m/s), and as unknown when a velocity component is missing
or not finite. `--stationary-speed 0.05` applies a separately declared slow/stationary
convention instead; in the two archived dumps it selects the same records.

Bearing, reactivity, robot speed and the heading-change axis are v1's functions, unchanged.
A classifier of true relative velocity would be a different analysis and is not this one.
The recorded robot state lags the pedestrians' by one step, which bounds how physically all
of these axes can be read.

Usage:  python3 collision_taxonomy_v2.py <trajectory_dir> [--json out.json]
"""
import argparse
import glob
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collision_taxonomy as v1  # noqa: E402

VERSION = 'collision_taxonomy_v2 2026-10-13'
STATIONARY_SPEED = 1e-8


def approach_direction(step, j, stationary_speed=STATIONARY_SPEED):
    """(sector, direction) for pedestrian j at this step; direction is one of
    oncoming / crossing / receding / stationary / unknown."""
    _, rpx, rpy = step[0], step[1], step[2]
    h = step[6][j]
    # Compute bearing before inspecting velocity; v1's motion classifier assumes hvy
    # exists whenever hvx exists and cannot safely be called on a partial velocity.
    bearing = abs((math.atan2(h[1] - rpy, h[0] - rpx) - step[5] + math.pi)
                  % (2 * math.pi) - math.pi)
    sector = ('front' if bearing < math.pi / 4 else
              ('side' if bearing < 3 * math.pi / 4 else 'behind'))
    hvx, hvy = h[2], h[3]
    if hvx is None or hvy is None or not (math.isfinite(hvx) and math.isfinite(hvy)):
        return sector, 'unknown'
    if math.hypot(hvx, hvy) <= stationary_speed:
        return sector, 'stationary'
    rel = math.atan2(h[1] - rpy, h[0] - rpx)
    hv_ang = math.atan2(hvy, hvx)
    d = abs(((hv_ang - rel + math.pi) % (2 * math.pi)) - math.pi)
    return sector, ('oncoming' if d > 2.0 else ('crossing' if d > 1.0 else 'receding'))


def classify(traj_dir, stationary_speed=STATIONARY_SPEED):
    rows = []
    for fn in sorted(glob.glob(os.path.join(traj_dir, 'ep_*.pkl'))):
        with open(fn, 'rb') as f:
            d = pickle.load(f)
        if not d.get('steps') or len(d['steps'][0]) < 7:
            continue
        if not str(d['outcome']).startswith('CollisionHuman'):
            continue
        last = d['steps'][-1]
        j, gap = v1.colliding_human(last, d['robot_radius'])
        if j is None:
            continue
        sector, direction = approach_direction(last, j, stationary_speed)
        h = last[6][j]
        usable_velocity = h[2] is not None and h[3] is not None and math.isfinite(h[2]) and math.isfinite(h[3])
        _, v1_kind = v1.approach_class(last, j) if usable_velocity else (sector, 'unknown')
        rows.append({
            'episode': d.get('episode'),
            'sector': sector,
            'direction': direction,
            'v1_kind': v1_kind,
            'reactive': bool(h[5]) if len(h) > 5 else None,
            'pedestrian_speed': math.hypot(h[2], h[3]) if usable_velocity else None,
            'robot_speed': math.hypot(last[3], last[4]) if last[3] is not None else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traj_dir')
    ap.add_argument('--stationary-speed', type=float, default=STATIONARY_SPEED)
    ap.add_argument('--json')
    a = ap.parse_args()
    rows = classify(a.traj_dir, a.stationary_speed)
    if not rows:
        sys.exit('no pedestrian collisions with usable trajectories in %s' % a.traj_dir)
    n = len(rows)

    def tally(key, order):
        c = {k: 0 for k in order}
        for r in rows:
            c[r[key]] = c.get(r[key], 0) + 1
        return c

    sectors = tally('sector', ('side', 'behind', 'front'))
    directions = tally('direction', ('oncoming', 'crossing', 'receding', 'stationary', 'unknown'))
    v1_kinds = tally('v1_kind', ('oncoming', 'crossing', 'receding', 'unknown'))
    nonreact = sum(1 for r in rows if r['reactive'] is False)
    sp = np.array([r['robot_speed'] for r in rows if r['robot_speed'] is not None])
    print('%s   pedestrian collisions: %d   stationary threshold: %g m/s'
          % (VERSION, n, a.stationary_speed))
    print('non-reactive colliding pedestrian: %d/%d (%.1f%%)' % (nonreact, n, 100.0 * nonreact / n))
    print('impact bearing: ' + ' / '.join('%s %d (%.1f%%)' % (k, v, 100.0 * v / n)
                                          for k, v in sectors.items()))
    print('pedestrian approach direction: ' + ' / '.join(
        '%s %d (%.1f%%)' % (k, v, 100.0 * v / n) for k, v in directions.items()))
    print('  (v1 "relative motion" for comparison: ' + ' / '.join(
        '%s %d' % (k, v) for k, v in v1_kinds.items()) + ')')
    moved = [(r['episode'], r['v1_kind']) for r in rows if r['direction'] in ('stationary', 'unknown')]
    print('  records reclassified as stationary/unknown (episode, v1 class): %s' % moved)
    print('robot recorded speed at impact: mean %.3f m/s, median %.3f m/s' % (sp.mean(), np.median(sp)))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump({'version': VERSION, 'traj_dir': os.path.abspath(a.traj_dir),
                       'stationary_speed': a.stationary_speed, 'n': n,
                       'non_reactive': nonreact, 'sectors': sectors, 'directions': directions,
                       'v1_kinds': v1_kinds, 'robot_speed_mean': float(sp.mean()),
                       'rows': rows}, f, indent=1)
        print('written to %s' % a.json)


if __name__ == '__main__':
    main()
