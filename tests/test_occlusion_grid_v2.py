"""build_label_grid_v2 (source/crowd_sim/envs/occlusion_grid.py) against an independent
brute-force reference, and against the historical build_label_grid where they must agree.

Reference convention (v1's snapped centre): with centre cell (cr, cc) = floor((l + half) /
res), possibly outside the grid, a cell (row, col) is occupied when
hypot(row - cr, col - cc) * res <= radius, for pedestrians not rejected by
|l| > half + radius.
"""
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'source', 'crowd_sim', 'envs'))
import occlusion_grid as og  # noqa: E402

RES, SIZE = 0.1, 100
HALF = SIZE * RES / 2


def sim_with(humans):
    return SimpleNamespace(config=SimpleNamespace(pas=SimpleNamespace(grid_res=RES, grid_size=SIZE)),
                           robot=SimpleNamespace(px=0.0, py=0.0, theta=0.0),
                           humans=[SimpleNamespace(px=x, py=y, radius=r) for x, y, r in humans])


def reference(humans):
    g = np.zeros((SIZE, SIZE), np.float32)
    rows, cols = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    s = sim_with([])
    for x, y, r in humans:
        # same robot-frame rotation as the production code, so cells on an exact boundary
        # fall on the same side; the rasterization below is independent of it
        x, y = og._world_to_ray_frame(s, x, y)
        if abs(x) > HALF + r or abs(y) > HALF + r:
            continue
        cc = math.floor((x + HALF) / RES)
        cr = math.floor((y + HALF) / RES)
        g[np.hypot(rows - cr, cols - cc) * RES <= r] = 1.0
    return g


def test_interior_pedestrians_match_v1():
    rnd = random.Random(3)
    for _ in range(300):
        humans = [(rnd.uniform(-4.5, 4.5), rnd.uniform(-4.5, 4.5), rnd.choice([0.25, 0.3, 0.15]))
                  for _ in range(rnd.randint(1, 8))]
        s = sim_with(humans)
        assert np.array_equal(og.build_label_grid(s), og.build_label_grid_v2(s))


def test_review_case_edge_pedestrian():
    s = sim_with([(5.1, 0.0, 0.3)])
    assert og.build_label_grid(s).sum() == 0                     # v1 drops it
    v2 = og.build_label_grid_v2(s)[0]
    assert v2.sum() > 0 and np.array_equal(v2, reference([(5.1, 0.0, 0.3)]))


def test_edges_corners_and_boundaries():
    cases = []
    for r in (0.05, 0.15, 0.25, 0.3, 0.4):
        for off in (-0.04, 0.0, 0.04, r - 0.01, r + 0.01):
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                x = sx * (HALF + off) if sx else 0.37
                y = sy * (HALF + off) if sy else -1.23
                cases.append((x, y, r))
    for h in cases:
        got = og.build_label_grid_v2(sim_with([h]))[0]
        assert np.array_equal(got, reference([h])), h


def test_random_near_edge_against_reference():
    rnd = random.Random(5)
    for _ in range(500):
        humans = [(rnd.choice([-1, 1]) * rnd.uniform(HALF - 0.5, HALF + 0.5), rnd.uniform(-HALF - 0.5, HALF + 0.5),
                   rnd.uniform(0.05, 0.45)) for _ in range(rnd.randint(1, 4))]
        assert np.array_equal(og.build_label_grid_v2(sim_with(humans))[0], reference(humans))


def test_no_intersection_writes_nothing():
    s = sim_with([(HALF + 0.31, 0.0, 0.3), (0.0, -HALF - 0.5, 0.25)])
    assert og.build_label_grid(s).sum() == 0 and og.build_label_grid_v2(s).sum() == 0


def test_build_grids_still_historical():
    import inspect
    src = inspect.getsource(og.build_grids)
    assert 'build_label_grid(sim)' in src and 'build_label_grid_v2' not in src


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
