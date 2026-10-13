"""stats/statlib.py against closed forms and standard reference values."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'stats'))
import statlib as S  # noqa: E402


def close(a, b, tol):
    return abs(a - b) <= tol


def test_t_distribution():
    assert close(S.t_sf(1.0, 1), 0.25, 1e-12)                                  # Cauchy
    assert close(S.t_sf(2.0, 2), 0.5 - 2 / (2 * math.sqrt(6)), 1e-12)          # closed form, df=2
    assert close(S.t_two_sided(2.0, 10), 0.0733880347707, 1e-10)
    assert close(S.t_ppf(0.975, 2), 4.302652729749464, 1e-8)
    assert close(S.t_ppf(0.975, 10), 2.228138851986274, 1e-8)
    assert close(S.t_two_sided(1.959963984540054, float('inf')), 0.05, 1e-12)


def test_exact_binomial_and_mcnemar():
    assert close(S.binom_cdf(3, 10), 176 / 1024, 1e-15)
    assert S.mcnemar_exact(0, 0) == 1.0 and S.mcnemar_exact(5, 5) == 1.0
    assert close(S.mcnemar_exact(0, 6), 2 / 64, 1e-15)                         # one-sided toy pair
    assert close(S.mcnemar_exact(195, 155), 0.03695105185874507, 1e-12)
    assert S.mcnemar_exact(1750, 1750) == 1.0                              # full cohort, no float overflow


def test_proportions():
    t = S.two_prop(3127, 3500, 3021, 3500)                                     # Table II, 3 m
    assert close(100 * t['diff'], 3.0286, 1e-3) and close(t['p'], 1.06e-4, 0.05e-4)
    lo, hi = S.wilson(335, 336)
    assert close(lo, 0.98334, 1e-4) and close(hi, 0.99947, 1e-4)
    assert close(S.power_two_prop(0.88, 0.90, 500), 0.1714, 1e-3)


def test_rank_test_and_effect_sizes():
    r = S.mann_whitney([1, 2, 3], [4, 5, 6])
    assert r['U'] == 0 and r['cliffs_delta'] == -1.0
    r = S.mann_whitney([1, 1, 2], [1, 1, 2])
    assert close(r['z'], 0.0, 1e-12) and close(r['cliffs_delta'], 0.0, 1e-12)
    assert close(S.cohen_h(0.5, 0.5), 0.0, 1e-15)


def test_holm():
    adj = S.holm([0.01, 0.04, 0.03, 0.2])
    assert [round(x, 4) for x in adj] == [0.04, 0.09, 0.09, 0.2]


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
