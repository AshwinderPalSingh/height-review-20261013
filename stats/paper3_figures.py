#!/usr/bin/env python3
"""Figures for the paper (version 2026-10-13 v2), drawn from stats/outputs/v2_2026-10-13/paper3_stats.json and,
for the collision example, from the primary trajectory dump.

    python3 stats/paper3_figures.py [--root .] [--primary DIR] [--stats JSON] [--out DIR]

Writes figures/ beside the versioned statistics JSON by default, preserving earlier outputs.
Writes fig_{four_axes,oracle_range,reciprocity_curve,collision_case}.
{pdf,png} with TrueType fonts (pdf.fonttype 42). Interval conventions, as in the captions:
  * four axes: perception, control and reciprocity bars are oracle/shield/reciprocity minus
    control success with the unpooled 95% CI over n=3,500 episodes per cell; the
    architecture bar is the three-seed mean minus HEIGHT with a t interval on the seed mean
    (df=2) -- three conditional fine-tuning seeds, not episodes.
  * oracle range: 95% CI of the oracle - occluded difference (unpooled, n=3,500 per cell).
  * reciprocity curve: 95% Wald interval of each success rate, n=3,500 per cell.
  * collision case: chosen automatically -- non-reactive pedestrian collision, side bearing,
    pedestrian approaching (v1 class "oncoming"), no heading change above 29 degrees in the
    record, at least 45 records, largest drop in recorded robot speed over the last 1.2 s.
"""
import argparse
import glob
import json
import math
import os
import pickle
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import statlib as S  # noqa: E402
from provenance import file_sha256, prepare_trajectories  # noqa: E402

N = 3500


def ci_pp(a, b, n=N):
    p1, p2 = a / n, b / n
    return 100 * 1.96 * math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)


def four_axes(st, out):
    ora = [x for x in st['oracle_range']['table'] if x['range_m'] == 25][0]
    arch = st['architecture']
    sh = st['shield']['cells']['ours']
    rec = st['reciprocity']
    o02, o10 = rec['table'][0]['ours_pct'], rec['table'][4]['ours_pct']
    k = lambda pc: round(pc * N / 100)  # noqa: E731
    gain = [ora['gap_pp'], arch['diff_pp'], sh['diff_pp'], o10 - o02]
    err = [ci_pp(k(ora['oracle_pct']), k(ora['occluded_pct'])),
           S.t_ppf(0.975, 2) * arch['sd_pp'] / math.sqrt(3),
           ci_pp(sh['success_on'], sh['success_off']),
           ci_pp(k(o10), k(o02))]
    labels = ['Perfect\nperception\n(25 m)', 'Architecture\n(intention+\nocclusion)',
              'Reactive\ncontrol (ped.-\nonly shield)', 'Reciprocity\n0.2 → 1.0']
    fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=300)
    ax.bar(range(4), gain, yerr=err, color=['#8c8c8c'] * 3 + ['#1f4e79'], capsize=3, width=0.6,
           error_kw=dict(lw=0.8))
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel('Δ success rate (pp)', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylim(-2.5, 12.0)
    for i, (g, e) in enumerate(zip(gain, err)):
        ax.text(i, g + e + 0.35, f'{g:+.2f}', ha='center', fontsize=6.5)
    ax.text(3, 8.4, '$p<10^{-68}$', ha='center', fontsize=6, color='white')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(out, 'fig_four_axes.' + ext))
    plt.close(fig)
    return {'bars_pp': gain, 'errors_pp': err}


def oracle_range(st, out):
    rows = st['oracle_range']['table']
    rng = np.array([x['range_m'] for x in rows])
    gap = np.array([x['gap_pp'] for x in rows])
    lo = np.array([x['ci_pp'][0] for x in rows])
    hi = np.array([x['ci_pp'][1] for x in rows])
    p3 = rows[0]['p']
    fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=300)
    ax.errorbar(rng, gap, yerr=[gap - lo, hi - gap], fmt='o-', color='#1f4e79', lw=1.5, ms=4.5, capsize=3)
    ax.axhline(0, color='k', lw=0.7)
    ax.axvline(25, color='k', ls=':', lw=0.9)
    ax.text(23.6, 3.9, 'benchmark\ndefault', fontsize=6.3, ha='right', va='top')
    mant, expo = ('%.1e' % p3).split('e')
    ax.annotate('$p=%s\\times10^{%d}$' % (mant, int(expo)), xy=(3.15, gap[0]), xytext=(6.0, 4.3), fontsize=6.3,
                arrowprops=dict(arrowstyle='-', lw=0.5, color='#666'))
    ax.set_xlabel('LiDAR sensing range (m)', fontsize=8)
    ax.set_ylabel('Oracle $-$ occluded (pp)', fontsize=8)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(mt.FixedLocator(rng))
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.xaxis.set_major_formatter(mt.FixedFormatter([str(v) for v in rng]))
    ax.set_xlim(2.6, 30)
    ax.set_ylim(-2, 5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(out, 'fig_oracle_range.' + ext))
    plt.close(fig)
    return {'gap_pp': gap.tolist(), 'ci_pp': [lo.tolist(), hi.tolist()]}


def reciprocity(st, out):
    rows = st['reciprocity']['table']
    vp = [x['visible_prob'] for x in rows]
    ours = [x['ours_pct'] for x in rows]
    hgt = [x['height_pct'] for x in rows]
    wald = lambda pc: 1.96 * math.sqrt((pc / 100) * (1 - pc / 100) / N) * 100  # noqa: E731
    oracle_default = [x for x in st['oracle_range']['table'] if x['range_m'] == 25][0]['oracle_pct']
    fig, ax = plt.subplots(figsize=(3.6, 2.7), dpi=300)
    ax.errorbar(vp, ours, yerr=[wald(c) for c in ours], marker='o', ms=4, lw=1.4, color='#2b5c8a', capsize=2.5,
                label='Ours (intention + occlusion)')
    ax.errorbar(vp, hgt, yerr=[wald(c) for c in hgt], marker='s', ms=4, lw=1.4, ls='--', color='#9a9a9a',
                capsize=2.5, label='HEIGHT')
    ax.axvline(0.2, color='k', ls=':', lw=0.9)
    ax.axhline(oracle_default, color='#2b5c8a', lw=0.7, alpha=0.35)
    ax.text(0.98, oracle_default + 0.35, 'perfect-perception oracle\nat default: no detectable gain',
            ha='right', va='bottom', fontsize=5.6, color='#2b5c8a')
    ax.text(0.215, 86.4, 'benchmark default', fontsize=6.2)
    ax.set_xlabel(r'$\mathtt{visible\_prob}$  (per-pedestrian reciprocity parameter)', fontsize=7.5)
    ax.set_ylabel('Success rate (%)', fontsize=8)
    ax.set_xticks(vp)
    ax.tick_params(labelsize=7)
    ax.set_ylim(86, 100)
    ax.legend(fontsize=6.4, frameon=False, loc='upper left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', alpha=0.22, lw=0.5)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(out, 'fig_reciprocity_curve.' + ext))
    plt.close(fig)
    return {'ours_pct': ours, 'height_pct': hgt, 'oracle_line_pct': oracle_default}


def collision_case(primary_dir, root, out):
    sys.path.insert(0, os.path.join(root, 'code'))
    import collision_taxonomy as ct
    eps = []
    for fn in sorted(glob.glob(os.path.join(primary_dir, 'ep_*.pkl'))):
        with open(fn, 'rb') as f:
            eps.append(pickle.load(f))
    cands = []
    for d in eps:
        if not str(d['outcome']).startswith('CollisionHuman') or len(d['steps']) < 45:
            continue
        st = d['steps'][-1]
        j, _ = ct.colliding_human(st, d['robot_radius'])
        if j is None or bool(st[6][j][5]):
            continue
        sector, kind = ct.approach_class(st, j)
        if sector != 'side' or kind != 'oncoming':
            continue
        if ct.steps_since_heading_change(d['steps'], j, math.radians(29)) is not None:
            continue
        if not d.get('obstacle_vertices'):
            continue
        n = len(d['steps'])
        v0 = math.hypot(st[3], st[4])
        v12 = math.hypot(*d['steps'][n - 1 - 12][3:5])
        cands.append((v12 - v0, d['episode'], v12, v0, d))
    cands.sort(key=lambda x: -x[0])
    _, ep, v12, v0, d = cands[0]
    steps = d['steps']
    n = len(steps)
    dt = d.get('time_step', 0.1)
    rr = d['robot_radius']
    j, _ = ct.colliding_human(steps[-1], rr)
    A = d.get('arena_size', 4.5)
    heading = [math.atan2(s[6][j][3], s[6][j][2]) for s in steps[n - 1 - 40:]]
    span = max(heading) - min(heading)
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0), dpi=300)
    for ax, tback in zip(axes, (4.0, 2.5, 1.2, 0.0)):
        s = steps[max(n - 1 - int(round(tback / dt)), 0)]
        for rect in d['obstacle_vertices']:
            a = np.asarray(rect, float).reshape(-1, 2)
            x0, y0, x1, y1 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='#d9d9d9', edgecolor='#8c8c8c', lw=0.6, zorder=1))
        for i, h in enumerate(s[6]):
            if h[0] is None:
                continue
            col, ec = ('#c0392b', '#7a1f16') if i == j else ('#bdbdbd', '#8c8c8c')
            ax.add_patch(Circle((h[0], h[1]), h[4], facecolor=col, edgecolor=ec, lw=0.6,
                                alpha=0.95 if i == j else 0.8, zorder=3))
            if math.hypot(h[2], h[3]) > 0.05:
                ax.arrow(h[0], h[1], h[2] * 0.6, h[3] * 0.6, head_width=0.09, color=ec if i != j else '#c0392b',
                         lw=0.8, zorder=4, length_includes_head=True)
        rx, ry, rvx, rvy, rth = s[1], s[2], s[3], s[4], s[5]
        ax.add_patch(Circle((rx, ry), rr, facecolor='#2b5c8a', edgecolor='#0f2a45', lw=0.8, zorder=5))
        ax.arrow(rx, ry, 0.45 * math.cos(rth), 0.45 * math.sin(rth), head_width=0.1, color='black', lw=1.0,
                 zorder=6, length_includes_head=True)
        v = math.hypot(rvx, rvy)
        ax.set_xlim(-A, A)
        ax.set_ylim(-A, A)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ttl = ('$t=0$,  $v$ = %.2f m/s   IMPACT' % v) if tback == 0 else ('$t=-%.1f$ s,  $v$ = %.2f m/s' % (tback, v))
        ax.set_title(ttl, fontsize=9, color='#c0392b' if tback == 0 else 'black')
        for sp_ in ax.spines.values():
            sp_.set_color('#555')
    plt.tight_layout(w_pad=0.6)
    for ext in ('pdf', 'png'):
        plt.savefig(os.path.join(out, 'fig_collision_case.' + ext))
    plt.close(fig)
    return {'episode': ep, 'candidates': len(cands), 'v_minus_1p2s': v12, 'v_impact': v0,
            'collider_heading_span_deg_last_4s': math.degrees(span)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--primary')
    ap.add_argument('--stats', help='Statistics JSON; default: versioned driver output')
    ap.add_argument('--out', help='Figure directory; default: figures/ beside statistics JSON')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    stats_path = a.stats or os.path.join(root, 'stats', 'outputs', 'v2_2026-10-13', 'paper3_stats.json')
    st = json.load(open(stats_path))
    for name, identity in st['inputs'].items():
        if file_sha256(os.path.join(root, name)) != identity['sha256']:
            raise ValueError('Statistics input has changed; rerun paper3_stats.py: ' + name)
    out = a.out or os.path.join(os.path.dirname(stats_path), 'figures')
    os.makedirs(out, exist_ok=True)
    archive = os.path.join(root, 'data', 'trajectories_n3500.tar.gz')
    with tempfile.TemporaryDirectory(prefix='paper3_figs_') as tmp:
        primary, manifest = prepare_trajectories(archive, tmp, a.primary)
        if manifest['traj_dir_sha256'] != st['trajectory_provenance']['primary']['traj_dir_sha256']:
            raise ValueError('Figure trajectories differ from the statistics cohort')
        rec = {'four_axes': four_axes(st, out), 'oracle_range': oracle_range(st, out),
               'reciprocity_curve': reciprocity(st, out), 'collision_case': collision_case(primary, root, out)}
    rec['provenance'] = {'version': 'paper3_figures 2026-10-13 v2',
        'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'inputs': {os.path.relpath(p, root): file_sha256(p) for p in
                   (stats_path, archive, __file__, os.path.join(HERE, 'statlib.py'),
                    os.path.join(HERE, 'provenance.py'), os.path.join(root, 'code', 'collision_taxonomy.py'))},
        'trajectory_manifest_sha256': manifest['traj_dir_sha256'], 'trajectory_files': manifest['n_files']}
    with open(os.path.join(out, 'figure_values.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
