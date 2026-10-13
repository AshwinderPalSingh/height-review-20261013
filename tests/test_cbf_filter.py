"""Fixtures for code/cbf_filter.py.

1. The default pedestrian-only variant (obstacles=False) behaves exactly as the filter that
   ran the reported 2x2: _legacy_min_clearance below is that version's method, copied
   verbatim, and both are compared on random scenes, with and without pedestrians.
2. obstacles=True scores rectangles, including when no pedestrian is present (the 13 Oct
   2026 fix); obstacles=False never does.
"""
import os
import random
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'code'))
import cbf_filter  # noqa: E402
from cbf_filter import CBFFilter, DELTAS  # noqa: E402


def _legacy_min_clearance(self, env, a, steps):
    """cbf_filter.py as it ran on 9-10 Sep 2026 (verbatim)."""
    humans = [(h.px, h.py, h.vx, h.vy, h.radius) for h in env.humans]
    if not humans:
        return None
    px, py, th = env.robot.px, env.robot.py, env.robot.theta
    v, w = env.desiredVelocity[0], env.desiredVelocity[1]
    dv, dw = DELTAS[a]
    worst = 1e9
    for t in range(1, steps + 1):
        px, py, th, v, w = self._roll(px, py, th, v, w, dv, dw)
        for (hx, hy, hvx, hvy, hr) in humans:
            fx, fy = hx + hvx * self.dt * t, hy + hvy * self.dt * t
            d = np.hypot(px - fx, py - fy) - (self.r_rob + hr)
            worst = min(worst, d)
        if self.obstacles:
            worst = min(worst, self._obstacle_clearance(env, px, py))
        if worst < -self.margin:
            break
    return worst


CONFIG = SimpleNamespace(env=SimpleNamespace(time_step=0.1),
                         robot=SimpleNamespace(v_min=-0.5, v_max=0.5, w_min=-1.0, w_max=1.0, radius=0.2))


def scene(humans, rects, x=0.0, y=0.0, th=0.0, v=0.4, w=0.0):
    env = SimpleNamespace(
        humans=[SimpleNamespace(px=a, py=b, vx=c, vy=d, radius=0.25) for a, b, c, d in humans],
        robot=SimpleNamespace(px=x, py=y, theta=th), desiredVelocity=[v, w])
    env.obstacle_coord = [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, y0, x1, y1 in rects]
    return env


def test_default_variant_unchanged_on_random_scenes():
    rnd = random.Random(11)
    for trial in range(400):
        humans = [(rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(-0.5, 0.5), rnd.uniform(-0.5, 0.5))
                  for _ in range(rnd.choice([0, 0, 1, 3, 6]))]
        rects = [(rnd.uniform(-2, 1), rnd.uniform(-2, 1), 0, 0) for _ in range(rnd.randint(0, 3))]
        rects = [(a, b, a + 0.6, b + 0.4) for a, b, _, _ in rects]
        env = scene(humans, rects, th=rnd.uniform(-3, 3), v=rnd.uniform(-0.5, 0.5), w=rnd.uniform(-1, 1))
        new = CBFFilter(CONFIG, horizon=10, margin=0.10)            # obstacles defaults to False
        old = CBFFilter(CONFIG, horizon=10, margin=0.10)
        old._min_clearance = _legacy_min_clearance.__get__(old)
        for a in range(9):
            assert new._min_clearance(env, a, 10) == old._min_clearance(env, a, 10), trial
            assert new.filter(env, a) == old.filter(env, a), trial


def test_obstacle_flag_changes_clearance_of_a_blocked_action():
    env = scene([(3.0, 3.0, 0.0, 0.0)], [(0.35, -0.5, 0.95, 0.5)], th=0.0, v=0.5)
    ahead = [k for k, d in DELTAS.items() if d == (0.05, 0.0)][0]
    ped_only = CBFFilter(CONFIG, horizon=10, margin=0.10, obstacles=False)
    aware = CBFFilter(CONFIG, horizon=10, margin=0.10, obstacles=True)
    assert ped_only._min_clearance(env, ahead, 10) > 3.0             # the rectangle is not scored
    assert aware._min_clearance(env, ahead, 10) < 0.0               # it is now
    assert ped_only.filter(env, ahead) == ahead                     # no override without obstacles
    assert aware.filter(env, ahead) != ahead


def test_no_pedestrian_scene():
    env = scene([], [(0.35, -0.5, 0.95, 0.5)], th=0.0, v=0.5)
    ahead = [k for k, d in DELTAS.items() if d == (0.05, 0.0)][0]
    assert CBFFilter(CONFIG, obstacles=False)._min_clearance(env, ahead, 10) is None
    c = CBFFilter(CONFIG, obstacles=True)._min_clearance(env, ahead, 10)
    assert c is not None and c < 0.0                                # before 13 Oct this returned None
    assert CBFFilter(CONFIG, obstacles=True).filter(env, ahead) != ahead


def test_docstring_names_the_variants():
    doc = cbf_filter.__doc__
    assert 'obstacles=False' in doc and 'pedestrians only' in doc and 'no safety guarantee' in doc


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
