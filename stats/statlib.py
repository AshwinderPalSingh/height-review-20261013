"""Statistical procedures used by paper3_stats.py. Standard library and NumPy only.

Conventions (as in the paper):
  * two_prop: difference a - b, 95% CI from the unpooled standard error, two-sided p from
    the pooled-variance z test.
  * welch: difference a - b, t with Welch-Satterthwaite degrees of freedom, two-sided p
    and 95% CI from Student's t (exact, not a normal approximation).
  * paired_t: mean of a - b over matched pairs, df = n - 1.
  * mcnemar_exact: 2 * P[Binomial(b + c, 1/2) <= min(b, c)], capped at 1.
  * wilson: Wilson score interval.
  * mann_whitney: U with average ranks, normal approximation with tie correction.
"""
import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_ppf(p):
    return _N.inv_cdf(p)


def _betacf(a, b, x, itmax=10000, eps=1e-15):
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError('betacf did not converge')


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student's t with df degrees of freedom (df may be fractional)."""
    if math.isinf(df):
        return norm_sf(t)
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return tail if t >= 0 else 1.0 - tail


def t_two_sided(t, df):
    return min(1.0, 2.0 * t_sf(abs(t), df))


def t_ppf(p, df):
    """Quantile of Student's t, by bisection on t_sf."""
    if math.isinf(df):
        return norm_ppf(p)
    if p == 0.5:
        return 0.0
    target = 1.0 - p
    lo, hi = -1e3, 1e3
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def two_prop(ka, na, kb, nb, conf=0.95):
    pa, pb = ka / na, kb / nb
    diff = pa - pb
    se_u = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    zc = norm_ppf(0.5 + conf / 2.0)
    pp = (ka + kb) / (na + nb)
    se_p = math.sqrt(pp * (1 - pp) * (1.0 / na + 1.0 / nb))
    z = diff / se_p if se_p > 0 else 0.0
    return {'pa': pa, 'pb': pb, 'diff': diff, 'ci': (diff - zc * se_u, diff + zc * se_u),
            'z': z, 'p': min(1.0, 2.0 * norm_sf(abs(z)))}


def welch_summary(ma, sa, na, mb, sb, nb, conf=0.95):
    va, vb = sa * sa / na, sb * sb / nb
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va * va / (na - 1) + vb * vb / (nb - 1))
    diff = ma - mb
    t = diff / se
    tc = t_ppf(0.5 + conf / 2.0, df)
    return {'diff': diff, 'se': se, 'df': df, 't': t, 'p': t_two_sided(t, df),
            'ci': (diff - tc * se, diff + tc * se)}


def welch(a, b, conf=0.95):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = welch_summary(a.mean(), a.std(ddof=1), a.size, b.mean(), b.std(ddof=1), b.size, conf)
    r.update({'mean_a': float(a.mean()), 'mean_b': float(b.mean()), 'n_a': int(a.size), 'n_b': int(b.size)})
    return r


def paired_t(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    se = d.std(ddof=1) / math.sqrt(n)
    t = d.mean() / se
    return {'diff': float(d.mean()), 'se': float(se), 'df': n - 1, 't': float(t), 'p': t_two_sided(t, n - 1)}


def binom_cdf(k, n, p=0.5):
    if p == 0.5:
        # Integer numerator avoids overflow when n reaches the 3,500-case cohort.
        return sum(math.comb(n, i) for i in range(0, min(k, n) + 1)) / (1 << n)
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(b, c), n, 0.5))


def wilson(k, n, conf=0.95):
    z = norm_ppf(0.5 + conf / 2.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = math.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return (a.mean() - b.mean()) / sp


def mann_whitney(a, b):
    """U for a (sum of a's ranks minus its minimum), z with tie correction, two-sided p,
    and Cliff's delta = P(a > b) - P(a < b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(allv.size)
    sv = allv[order]
    i = 0
    tie_term = 0.0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = r
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    ra = ranks[:na].sum()
    u = ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    n = na + nb
    sigma = math.sqrt(na * nb / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u - mu) / sigma
    delta = 2.0 * u / (na * nb) - 1.0
    return {'U': u, 'z': z, 'p': min(1.0, 2.0 * norm_sf(abs(z))), 'cliffs_delta': delta}


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1])


def tost_welch(diff, se_a, df_a, se_b, df_b, margin, conf=0.90):
    """Two one-sided tests on a difference whose variance has two components, each with its
    own degrees of freedom (Welch-Satterthwaite). Returns the (1-2alpha) CI and p = max of
    the two one-sided p-values."""
    se = math.sqrt(se_a ** 2 + se_b ** 2)
    df = (se_a ** 2 + se_b ** 2) ** 2 / ((se_a ** 4) / df_a + (0.0 if math.isinf(df_b) else (se_b ** 4) / df_b))
    tc = t_ppf(0.5 + conf / 2.0, df)
    p_low = t_sf((diff + margin) / se, df)        # H0: diff <= -margin
    p_high = t_sf((margin - diff) / se, df)       # H0: diff >= +margin
    return {'diff': diff, 'se': se, 'df': df, 'ci': (diff - tc * se, diff + tc * se),
            'p': max(p_low, p_high), 'p_low': p_low, 'p_high': p_high}


def power_two_prop(p1, p2, n, alpha=0.05):
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    return 1.0 - _N.cdf(norm_ppf(1 - alpha / 2.0) - abs(p1 - p2) / se)


def n_for_power(p1, p2, power=0.80, alpha=0.05):
    za, zb = norm_ppf(1 - alpha / 2.0), norm_ppf(power)
    return (za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2


def expected_max_normal(k, grid=200001, lo=-10.0, hi=10.0):
    """E[max of k iid standard normals], by numerical integration."""
    x = np.linspace(lo, hi, grid)
    pdf = np.exp(-x * x / 2.0) / math.sqrt(2 * math.pi)
    cdf = np.array([_N.cdf(v) for v in x[::100]])
    cdf = np.interp(x, x[::100], cdf)
    dens = k * cdf ** (k - 1) * pdf
    return float(np.trapezoid(x * dens, x) if hasattr(np, 'trapezoid') else np.trapz(x * dens, x))


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj
