"""code/collision_taxonomy_v2.py: pedestrian approach direction, stationary and unknown
classes, and agreement with v1 wherever the pedestrian moves."""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'code'))
import collision_taxonomy as v1  # noqa: E402
import collision_taxonomy_v2 as v2  # noqa: E402


def step(hx, hy, hvx, hvy, rvx=0.0, rvy=0.0, rth=0.0):
    return (0.0, 0.0, 0.0, rvx, rvy, rth, [(hx, hy, hvx, hvy, 0.25, False)])


def test_stationary_and_unknown():
    assert v2.approach_direction(step(1.0, 0.0, 0.0, 0.0), 0)[1] == 'stationary'
    assert v2.approach_direction(step(1.0, 0.0, 1e-9, 0.0), 0)[1] == 'stationary'
    assert v2.approach_direction(step(1.0, 0.0, 0.03, 0.0), 0)[1] == 'receding'        # 1e-8 rule
    assert v2.approach_direction(step(1.0, 0.0, 0.03, 0.0), 0, 0.05)[1] == 'stationary'
    assert v2.approach_direction(step(1.0, 0.0, float('nan'), 0.0), 0)[1] == 'unknown'
    for vx, vy in [(None, None), (0.1, None), (None, 0.1), (0.1, float('nan')),
                   (float('inf'), 0.0), (0.0, float('-inf'))]:
        assert v2.approach_direction(step(1.0, 0.0, vx, vy), 0)[1] == 'unknown'


def test_thresholds_and_robot_velocity_not_subtracted():
    # pedestrian ahead walking away at 0.2 m/s while the robot closes at 0.5 m/s: still
    # "receding" -- the axis is the pedestrian's own direction, as the new name says
    assert v2.approach_direction(step(1.0, 0.0, 0.2, 0.0, rvx=0.5), 0)[1] == 'receding'
    assert v2.approach_direction(step(1.0, 0.0, -0.5, 0.0), 0)[1] == 'oncoming'
    assert v2.approach_direction(step(1.0, 0.0, 0.0, 0.5), 0)[1] == 'crossing'
    for ang, want in ((2.0, 'crossing'), (1.0, 'receding'),
                      (2.0 + 1e-6, 'oncoming'), (2.0 - 1e-6, 'crossing'), (1.0 + 1e-6, 'crossing'),
                      (1.0 - 1e-6, 'receding')):
        hv = (0.5 * math.cos(ang), 0.5 * math.sin(ang))
        assert v2.approach_direction(step(1.0, 0.0, *hv), 0)[1] == want, ang


def test_matches_v1_for_moving_pedestrians():
    rnd = random.Random(9)
    for _ in range(2000):
        s = step(rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-0.6, 0.6), rnd.uniform(-0.6, 0.6),
                 rth=rnd.uniform(-3, 3))
        if math.hypot(s[6][0][2], s[6][0][3]) <= 1e-8:
            continue
        assert v2.approach_direction(s, 0) == v1.approach_class(s, 0)


def test_classify_missing_velocity_does_not_crash():
    import pickle
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        for i, (vx, vy) in enumerate([(0.1, None), (None, 0.1), (0.0, float('nan'))]):
            d = {'episode': i, 'outcome': 'CollisionHuman', 'robot_radius': 0.2,
                 'steps': [step(0.4, 0.0, vx, vy)]}
            with (Path(directory) / ('ep_%d.pkl' % i)).open('wb') as f:
                pickle.dump(d, f)
        rows = v2.classify(directory)
        assert len(rows) == 3
        assert all(r['direction'] == 'unknown' and r['pedestrian_speed'] is None for r in rows)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
