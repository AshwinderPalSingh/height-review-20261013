"""Fixtures for code/ics_analysis_v2.py (closed-prefix inevitable-collision search).

Each fixture is a tiny synthetic record in the dump format:
step = (t, rx, ry, rvx, rvy, rtheta, humans), human = (x, y, vx, vy, radius, reacts).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'code'))
import ics_analysis as v1  # noqa: E402
import ics_analysis_v2 as v2  # noqa: E402

R_ROB, R_HUM, DT = 0.20, 0.25, 0.1


def record(robot, humans, k):
    """k+1 identical samples: robot (x, y, theta, v) and fixed humans [(x, y)]."""
    x, y, th, v = robot
    import math
    hs = [(hx, hy, 0.0, 0.0, R_HUM, False) for hx, hy in humans]
    return [(i * DT, x, y, v * math.cos(th), v * math.sin(th), th, hs) for i in range(k + 1)]


def test_contact_then_separation_is_initial_contact():
    # Review fixture: start separation 0.44 m against a 0.45 m envelope, robot moving away.
    steps = record((0.0, 0.0, 0.0, 0.5), [(-0.44, 0.0)], 1)
    v1.PAD = 0.0
    assert v1.hits_human(0.0, 0.0, steps[0][6], R_ROB)
    assert v1.escapable(steps, 0, 1, R_ROB, 0.5, DT, None) is True       # historical semantics
    r = v2.search(steps, 0, 1, R_ROB, DT, None, 0.0)
    assert r['status'] == 'initial_contact' and r['initial_contact']['human'] == 0
    assert r['open_start'] == 'survives_horizon'                      # v1's answer is kept visible


def test_exact_boundary_counts_as_contact():
    steps = record((0.0, 0.0, 0.0, 0.0), [(R_ROB + R_HUM, 0.0)], 2)
    assert v2.search(steps, 0, 2, R_ROB, DT, None, 0.0)['status'] == 'initial_contact'
    steps = record((0.0, 0.0, 0.0, 0.0), [(R_ROB + R_HUM + 1e-6, 0.0)], 2)
    assert v2.search(steps, 0, 2, R_ROB, DT, None, 0.0)['status'] != 'initial_contact'


def test_rectangle_start_contact_and_pad():
    rect = [[0.25, -1.0], [1.25, -1.0], [1.25, 1.0], [0.25, 1.0]]     # 0.05 m beyond the robot surface
    steps = record((0.0, 0.0, 3.14159, 0.0), [(-3.0, 3.0)], 3)
    assert v2.search(steps, 0, 3, R_ROB, DT, [rect], 0.0)['status'] == 'survives_horizon'
    r = v2.search(steps, 0, 3, R_ROB, DT, [rect], 0.10)               # pad swallows the gap
    assert r['status'] == 'initial_contact' and r['initial_contact']['obstacle'] == 0


def test_zero_horizon():
    clear = record((0.0, 0.0, 0.0, 0.0), [(2.0, 0.0)], 0)
    hit = record((0.0, 0.0, 0.0, 0.0), [(0.3, 0.0)], 0)
    v1.PAD = 0.0
    assert v1.escapable(hit, 0, 0, R_ROB, 0.5, DT, None) is True      # v1 ignores contact at k=0
    assert v2.search(clear, 0, 0, R_ROB, DT, None, 0.0)['status'] == 'survives_horizon'
    assert v2.search(hit, 0, 0, R_ROB, DT, None, 0.0)['status'] == 'initial_contact'


def test_initially_clear_escape_has_a_replayable_witness():
    steps = record((0.0, 0.0, 0.0, 0.5), [(0.8, 0.0)], 10)            # head-on, 0.35 m of free space
    r = v2.search(steps, 0, 10, R_ROB, DT, None, 0.0)
    assert r['status'] == 'survives_horizon' and len(r['witness']) == 10
    assert v2.replay_witness(steps, 0, r['witness'], R_ROB, DT, None, 0.0)
    full_ahead = v2.DELTAS.index((0.05, 0.0))
    assert not v2.replay_witness(steps, 0, [full_ahead] * 10, R_ROB, DT, None, 0.0)


def test_boxed_in_has_no_surviving_sequence():
    ring = [(0.47 * __import__('math').cos(a / 8 * 6.2832), 0.47 * __import__('math').sin(a / 8 * 6.2832)) for a in range(8)]
    steps = record((0.0, 0.0, 0.0, 0.5), ring, 5)
    r = v2.search(steps, 0, 5, R_ROB, DT, None, 0.0)
    assert r['status'] == 'no_surviving_sequence', r['status']


def test_v2_matches_v1_when_start_is_clear():
    import random
    rnd = random.Random(7)
    for _ in range(200):
        humans = [(rnd.uniform(-1.5, 1.5), rnd.uniform(-1.5, 1.5)) for _ in range(rnd.randint(1, 5))]
        steps = record((0.0, 0.0, rnd.uniform(-3, 3), rnd.uniform(-0.5, 0.5)), humans, 6)
        pad = rnd.choice([0.0, 0.049, 0.1])
        v1.PAD = pad
        r = v2.search(steps, 0, 6, R_ROB, DT, None, pad)
        old = v1.escapable(steps, 0, 6, R_ROB, 0.5, DT, None)
        if r['status'] == 'initial_contact':
            assert (r['open_start'] == 'survives_horizon') == old
        else:
            assert (r['status'] == 'survives_horizon') == old


def test_aborted_search_has_an_explicit_status():
    steps = record((0.0, 0.0, 0.0, 0.0), [(3.0, 3.0)], 2)
    cap = v2.STATE_CAP
    try:
        v2.STATE_CAP = 0
        result = v2.search(steps, 0, 2, R_ROB, DT, None, 0.0)
        assert result['status'] == 'aborted' and result['witness'] is None
        hit = record((0.0, 0.0, 0.0, 0.0), [(0.4, 0.0)], 2)
        assert v2.search(hit, 0, 2, R_ROB, DT, None, 0.0)['status'] == 'initial_contact'
    finally:
        v2.STATE_CAP = cap


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
