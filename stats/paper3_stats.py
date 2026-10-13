#!/usr/bin/env python3
"""Statistics driver for the paper (version 2026-10-13).

Recomputes, from the archived inputs in this repository, the statistics the paper prints in
its tables, figures and text, and records the SHA-256 of every input it reads:

    stats/outputs/v2_2026-10-13/paper3_stats.json   values, inputs, cohorts and test designs
    stats/outputs/v2_2026-10-13/paper3_stats.md     the same, as readable tables

Run from the repository root (CPU only, standard library and NumPy):

    python3 stats/paper3_stats.py
    python3 stats/paper3_figures.py     # figures, from the versioned paper3_stats.json

The two trajectory dumps are extracted from data/*.tar.gz into a temporary directory unless
--primary / --repeat point at extracted copies; supplied copies must match every archived
pickle byte. Versioned outputs default to stats/outputs/v2_2026-10-13/. The inevitable-collision table is read from
results/v2/ics_v2_*.json, written by code/ics_analysis_v2.py; the frozen v1 outputs
(results/logs/ics_exact_results.txt, ics_full_results.json) are reported for comparison only.

Designs (procedures in stats/statlib.py):
  * Success rates: two-proportion z test (pooled SE for p, unpooled 95% CI).
  * Cross-checkpoint contrasts (ours: environment seeds 1-3; HEIGHT: 50569) are unpaired.
  * Contrasts within one checkpoint (oracle on/off, shield on/off, HEIGHT vs its own
    continuation) replay the same seed/index sequence. They are reported unpaired, and
    where stated also as exact McNemar or paired t on episode-index-matched outcomes. That
    pairing rests on the shared reset design; initial-scene hashes were not archived.
  * Continuous per-episode metrics: Welch t, Satterthwaite df, exact t tail.
  * Exploratory: the TOST (margin 1.1 pp, not fixed in advance) and the shield's
    per-collision-type McNemar tests (family of four, Bonferroni and Holm both given).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import statlib as S  # noqa: E402
from provenance import prepare_trajectories  # noqa: E402

VERSION = 'paper3_stats 2026-10-13 v2'
OUTPUT_VERSION = 'v2_2026-10-13'
N = 3500
INPUTS = {}
ROOT = None


# ----------------------------------------------------------------------------- inputs
def use(path, role):
    """Register an input file and return its path."""
    rel = os.path.relpath(path, ROOT)
    if rel not in INPUTS:
        with open(path, 'rb') as f:
            data = f.read()
        INPUTS[rel] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'roles': []}
    if role not in INPUTS[rel]['roles']:
        INPUTS[rel]['roles'].append(role)
    return path


def log(name, role):
    return use(os.path.join(ROOT, 'results', 'logs', name), role)


OUTCOME_LINES = {'Success': 'success', 'Collision with human': 'human',
                 'Collision with obstacle': 'obstacle', 'Collision with wall': 'wall',
                 'Time out': 'timeout'}


def parse_eval(name, role):
    """Per-episode outcomes and summaries of one evaluation log, cross-checked against the
    log's own final case lists."""
    path = log(name, role)
    with open(path) as f:
        lines = f.read().split('\n')
    n_eval = None
    ep_out, ep_steps = {}, {}
    cur = None
    cases = {}
    nav = intr = cbf = None
    totals = checkpoint = None
    for ln in lines:
        m = re.match(r'Episode (\d+) ends in (\d+)', ln)
        if m:
            if cur is not None:
                raise ValueError('%s: episode %d has no outcome' % (name, cur[0]))
            cur = (int(m.group(1)), int(m.group(2)))
            if cur[0] in ep_out:
                raise ValueError('%s: duplicate episode %d' % (name, cur[0]))
            continue
        s = ln.strip()
        if cur is not None and s in OUTCOME_LINES:
            ep_out[cur[0]] = OUTCOME_LINES[s]
            ep_steps[cur[0]] = cur[1]
            cur = None
            continue
        if cur is not None and s:
            raise ValueError('%s: episode %d has an unrecognized outcome: %s' % (name, cur[0], s))
        m = re.fullmatch(r'(\d+) (\d+) (\d+)', s)
        if m:
            totals = tuple(map(int, m.groups()))
        if re.fullmatch(r'data/[^/]+/checkpoints/[^/]+\.(pt|pth)', s):
            checkpoint = s
        m = re.search(r'Collision cases:(.*)$', ln)
        if m:
            cases['any_collision'] = [int(x) for x in m.group(1).split()]
        m = re.search(r'Collision with (Human|Obstacle|Wall) cases:(.*)$', ln)
        if m:
            cases[m.group(1).lower()] = [int(x) for x in m.group(2).split()]
        m = re.search(r'Timeout cases:(.*)$', ln)
        if m:
            cases['timeout'] = [int(x) for x in m.group(1).split()]
        m = re.search(r'nav time \(successes\): ([\d.]+) \+/- ([\d.]+) over n=(\d+)', ln)
        if m:
            nav = (float(m.group(1)), float(m.group(2)), int(m.group(3)))
        m = re.search(r'average intrusion ratio: ([\d.]+)', ln)
        if m:
            intr = float(m.group(1))
        m = re.search(r'CBF: (\d+) steps checked, (\d+) overridden \(([\d.]+)%\)', ln)
        if m:
            cbf = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
        m = re.search(r'Evaluation using (\d+) episodes', ln)
        if m:
            n_eval = int(m.group(1))
    n = n_eval
    assert cur is None, (name, 'incomplete final episode')
    assert n is not None and len(ep_out) == n and sorted(ep_out) == list(range(n)), name
    outcome = [ep_out[i] for i in range(n)]
    for k, key in (('human', 'human'), ('obstacle', 'obstacle'), ('wall', 'wall'), ('timeout', 'timeout')):
        listed = sorted(cases.get(k, []))
        printed = [i for i in range(n) if outcome[i] == key]
        assert listed == printed, (name, k)
    succ = sum(1 for o in outcome if o == 'success')
    collisions = [i for i in range(n) if outcome[i] in ('human', 'obstacle', 'wall')]
    assert cases.get('any_collision') is None or sorted(cases['any_collision']) == collisions, (name, 'total collision list')
    computed_totals = (succ, len(collisions), sum(o == 'timeout' for o in outcome))
    assert totals is None or totals == computed_totals, (name, 'count summary', totals, computed_totals)
    assert nav is None or nav[2] == succ, name
    nav_ep = np.array([(ep_steps[i] - 1) * 0.1 for i in range(n) if outcome[i] == 'success'])
    return {'name': name, 'n': n, 'outcome': outcome, 'succ': succ,
            'counts': {k: sum(1 for o in outcome if o == k) for k in OUTCOME_LINES.values()},
            'nav_summary': nav, 'nav_ep': nav_ep, 'intrusion_summary': intr, 'cbf': cbf,
            'checkpoint': checkpoint, 'validation': {'unique_complete_ids': True,
              'type_case_lists_match': True, 'aggregate_case_list_checked': 'any_collision' in cases,
              'count_summary_checked': totals is not None}}


def intrusion(name, role):
    path = log(name, role)
    return np.array(json.load(open(path))['too_close_ratios'], float)


def pct(k, n=N):
    return 100.0 * k / n


def r(x, nd):
    return None if x is None else round(float(x), nd)


def tp(a, b, label=None):
    """two_prop on parsed logs, a - b, in percentage points."""
    t = S.two_prop(a['succ'], a['n'], b['succ'], b['n'])
    return {'a': a['name'], 'b': b['name'], 'rate_a': 100 * t['pa'], 'rate_b': 100 * t['pb'],
            'diff_pp': 100 * t['diff'], 'ci_pp': [100 * t['ci'][0], 100 * t['ci'][1]],
            'z': t['z'], 'p': t['p']}


def success_vector(ev):
    return np.array([1 if o == 'success' else 0 for o in ev['outcome']])


# ----------------------------------------------------------------------------- dumps
def load_dump(d):
    eps = []
    for fn in sorted(glob.glob(os.path.join(d, 'ep_*.pkl'))):
        with open(fn, 'rb') as f:
            eps.append(pickle.load(f))
    return eps


# ----------------------------------------------------------------------------- sections
def reproduction():
    rows = [('HEIGHT-noAttn', 'base_HEIGHT_noAttn.log', 52), ('DS-RNN', 'base_DS_RNN.log', 80),
            ('HEIGHT-RH', 'base_HEIGHT_RH.log', 85), ('HomoGAT', 'base_HomoGAT.log', 86),
            ('HEIGHT-HH', 'base_HEIGHT_HH.log', 87), ('HEIGHT (full)', 'eval_heightpub.log', 88),
            ('HEIGHT, 10-14 pedestrians', 'gen_more_human_heightpub.log', 78),
            ('HEIGHT, 13-17 obstacles', 'gen_more_obs_heightpub.log', 84)]
    out = []
    for label, fn, pub in rows:
        ev = parse_eval(fn, 'Table I')
        out.append({'method': label, 'log': fn, 'reproduced_pct': pct(ev['succ']), 'published_pct': pub,
                    'delta_pp': pct(ev['succ']) - pub})
    ours_shift = {lab: pct(parse_eval(fn, 'Sec. IV text')['succ'])
                  for lab, fn in (('10-14 pedestrians', 'gen_more_human_nospeak.log'),
                                  ('13-17 obstacles', 'gen_more_obs_nospeak.log'))}
    gap = out[5]['reproduced_pct'] - out[0]['reproduced_pct']
    return {'table': out, 'ours_under_shifts_pct': ours_shift, 'ablation_gap_pp': gap,
            'design': 'success rate of each released checkpoint, n=3500; published values from the HEIGHT paper (n=500)'}


def protocol():
    p_ours, p_height = 0.88, 0.90
    res = {'power_n3500_2pp': S.power_two_prop(p_ours, p_height, 3500),
           'n_for_80pct_power': S.n_for_power(p_ours, p_height),
           'power_n500_2pp': S.power_two_prop(p_ours, p_height, 500),
           'best_of_five_bias_se': S.expected_max_normal(5),
           'design': 'two-sided two-proportion z, alpha 0.05, rates 0.88 vs 0.90; best-of-five bias is E[max of 5 standard normals]'}
    # seeding overlap: never-trained (first 1,000) vs once-seen (remaining 2,500) episodes
    ev = parse_eval('eval_exp_nospeak_s1.log', 'Sec. III seeding check')
    s = success_vector(ev)
    a, b = int(s[:1000].sum()), int(s[1000:].sum())
    t = S.two_prop(a, 1000, b, 2500)
    res['seeding_overlap'] = {'never_trained_pct': 100 * a / 1000, 'once_seen_pct': 100 * b / 2500, 'p': t['p']}
    # run-to-run variation of the frozen seed-1 checkpoint at the default configuration
    ours_default = ['eval_exp_nospeak_s1.log', 'soc_ours.log', 'ics_redump_eval.log', 'rc_rc25_occl.log',
                    're_vp02_ours.log', 'x_ours_off.log', 'ics_boost_eval.log', 'eval_traj_nospeak.log']
    evs = [parse_eval(fn, 'Sec. III run-to-run variation') for fn in ours_default]
    succ = {e['name']: e['succ'] for e in evs}
    agree = [e for e in evs if e['succ'] == 3099]
    identical = all(e['outcome'] == agree[0]['outcome'] for e in agree)
    height_default = ['eval_heightpub.log', 'soc_height.log', 're_vp02_height.log', 'x_height_off.log']
    res['run_to_run'] = {'ours_successes': succ, 'spread_pp': pct(max(succ.values()) - min(succ.values())),
                         'n_runs_3099': len(agree), 'runs_3099_identical_outcomes': identical,
                         'height_successes': {fn: parse_eval(fn, 'Sec. III run-to-run variation')['succ']
                                              for fn in height_default}}
    o = intrusion('intrusion_test_re_vp02_ours.json', 'Sec. III pairing check')
    h = intrusion('intrusion_test_re_vp02_height.json', 'Sec. III pairing check')
    res['intrusion_corr_ours_vs_height'] = S.pearson(o, h)
    return res


def oracle_range():
    rows = []
    for rng in (3, 5, 8, 15, 25):
        occ = parse_eval('rc_rc%d_occl.log' % rng, 'Table II')
        ora = parse_eval('rc_rc%d_oracle.log' % rng, 'Table II')
        t = tp(ora, occ)
        rows.append({'range_m': rng, 'occluded_pct': t['rate_b'], 'oracle_pct': t['rate_a'],
                     'gap_pp': t['diff_pp'], 'ci_pp': t['ci_pp'], 'p': t['p'],
                     'survives_bonferroni_5': t['p'] < 0.05 / 5})
    gaps = [x['gap_pp'] for x in rows]
    rngs = [x['range_m'] for x in rows]
    occ25 = success_vector(parse_eval('rc_rc25_occl.log', 'Sec. V discordance'))
    ora25 = success_vector(parse_eval('rc_rc25_oracle.log', 'Sec. V discordance'))
    pilot = []
    for rng in (3, 5, 8, 15, 25):
        o1 = parse_eval('rs_r%d_occl.log' % rng, 'Sec. XIII n=1000 range pilot')
        o2 = parse_eval('rs_r%d_oracle.log' % rng, 'Sec. XIII n=1000 range pilot')
        pilot.append({'range_m': rng, 'gap_pp': 100.0 * (o2['succ'] / o2['n'] - o1['succ'] / o1['n'])})
    pg = [x['gap_pp'] for x in pilot]
    return {'table': rows,
            'monotone_decreasing': all(gaps[i] >= gaps[i + 1] for i in range(len(gaps) - 1)),
            'pearson_range_gap': S.pearson(rngs, gaps),
            'pearson_logrange_gap': S.pearson(np.log(rngs), gaps),
            'default_25m': {'oracle_only_successes': int(((ora25 == 1) & (occ25 == 0)).sum()),
                            'occluded_only_successes': int(((ora25 == 0) & (occ25 == 1)).sum()),
                            'mcnemar_p': S.mcnemar_exact(int(((ora25 == 1) & (occ25 == 0)).sum()),
                                                         int(((ora25 == 0) & (occ25 == 1)).sum()))},
            'pilot_n1000': {'table': pilot, 'monotone_decreasing': all(pg[i] >= pg[i + 1] for i in range(len(pg) - 1))},
            'design': 'oracle minus occluded success, frozen seed-1 checkpoint, unpaired two-proportion z; Bonferroni threshold 0.05/5; 25 m discordance by episode index'}


def architecture():
    seeds = [parse_eval('eval_exp_nospeak_s%d.log' % s, 'Table III') for s in (1, 2, 3)]
    height = parse_eval('eval_heightpub.log', 'Table III')
    rates = np.array([pct(e['succ']) for e in seeds])
    mean, sd = rates.mean(), rates.std(ddof=1)
    h = pct(height['succ'])
    se_seed = sd / math.sqrt(3)
    tcrit = S.t_ppf(0.975, 2)
    t = (mean - h) / se_seed
    se_h = 100 * math.sqrt(height['succ'] / N * (1 - height['succ'] / N) / N)
    tost = S.tost_welch(mean - h, se_seed, 2, se_h, math.inf, 1.1)
    return {'seed_pct': rates.tolist(), 'mean_pct': mean, 'sd_pp': sd,
            'ci_mean_pct': [mean - tcrit * se_seed, mean + tcrit * se_seed],
            'height_pct': h, 'diff_pp': mean - h, 't': t, 'df': 2, 'p': S.t_two_sided(t, 2),
            'spread_pp': rates.max() - rates.min(),
            'tost': {'margin_pp': 1.1, 'ci90_pp': list(tost['ci']), 'p': tost['p'], 'df': tost['df']},
            'design': 'three fine-tuning seeds vs the released checkpoint; one-sample t on the seed mean (df=2); exploratory TOST combining the seed SE (df=2) and the binomial SE of HEIGHT (Welch-Satterthwaite df)'}


def shield():
    cells = {k: parse_eval('x_%s.log' % k, 'Table V') for k in ('height_off', 'height_on', 'ours_off', 'ours_on')}
    out = {'design': ('pedestrian-only shield (cbf.obstacles not set; default False), horizon 10 steps, '
                      'margin 0.10 m; off/on paired by episode index within each checkpoint (shared reset '
                      'seed/index design; initial-scene hashes not archived); HEIGHT and ours unpaired. '
                      'Per-type tests are exploratory: family of four (two checkpoints x pedestrian/obstacle).'),
           'cells': {}}
    type_tests = []
    for arch in ('height', 'ours'):
        off, on = cells[arch + '_off'], cells[arch + '_on']
        if off['checkpoint'] != on['checkpoint'] or off['checkpoint'] is None:
            raise ValueError('Shield off/on checkpoint mismatch: ' + arch)
        run_name = off['checkpoint'].split('/')[1]
        config = use(os.path.join(ROOT, 'configs', 'runs', run_name, 'config.py'), 'shield archived seed evidence')
        with open(config) as f:
            seed = int(re.search(r'env\.seed\s*=\s*(\d+)', f.read()).group(1))
        assert off['n'] == on['n'] == N, (arch, 'shield cohort size')
        t = tp(on, off)
        rec = {'success_off': off['succ'], 'success_on': on['succ'], 'rate_off_pct': t['rate_b'],
               'rate_on_pct': t['rate_a'], 'diff_pp': t['diff_pp'], 'p_unpaired': t['p'],
               'counts_off': off['counts'], 'counts_on': on['counts'],
               'fire_rate_pct': on['cbf'][2] if on['cbf'] else None, 'checked_overridden': on['cbf'][:2] if on['cbf'] else None,
               'paired': {}, 'pairing_evidence': {'logged_checkpoint_path': off['checkpoint'],
                 'archived_seed': seed,
                 'validation_off': off['validation'], 'validation_on': on['validation'],
                 'qualification': 'Shared checkpoint path and archived seed/index reset design; no archived scene or run-time checkpoint hashes.'}}
        for k in ('success', 'human', 'obstacle', 'wall', 'timeout'):
            a = np.array([o == k for o in off['outcome']])
            b = np.array([o == k for o in on['outcome']])
            b_only, a_only = int((b & ~a).sum()), int((a & ~b).sum())
            pu = S.two_prop(int(b.sum()), N, int(a.sum()), N)['p']
            rec['paired'][k] = {'off': int(a.sum()), 'on': int(b.sum()), 'off_only': a_only, 'on_only': b_only,
                                'off_only_ids': np.flatnonzero(a & ~b).tolist(), 'on_only_ids': np.flatnonzero(b & ~a).tolist(),
                                'mcnemar_p': S.mcnemar_exact(a_only, b_only), 'unpaired_p': pu}
            if k in ('human', 'obstacle'):
                type_tests.append((arch, k, S.mcnemar_exact(a_only, b_only)))
        anyc = lambda ev: np.array([o in ('human', 'obstacle', 'wall') for o in ev['outcome']])  # noqa: E731
        a, b = anyc(off), anyc(on)
        rec['paired']['any_collision'] = {'off': int(a.sum()), 'on': int(b.sum()),
                                          'off_only': int((a & ~b).sum()), 'on_only': int((b & ~a).sum()),
                                          'off_only_ids': np.flatnonzero(a & ~b).tolist(), 'on_only_ids': np.flatnonzero(b & ~a).tolist(),
                                          'mcnemar_p': S.mcnemar_exact(int((a & ~b).sum()), int((b & ~a).sum()))}
        trans = {}
        kinds = tuple(OUTCOME_LINES.values())
        matrix = {x: {y: 0 for y in kinds} for x in kinds}
        transition_ids = {x: {y: [] for y in kinds} for x in kinds}
        for i, (x, y) in enumerate(zip(off['outcome'], on['outcome'])):
            matrix[x][y] += 1
            transition_ids[x][y].append(i)
            if x != y:
                trans['%s->%s' % (x, y)] = trans.get('%s->%s' % (x, y), 0) + 1
        assert all(sum(matrix[x].values()) == off['counts'][x] for x in kinds), arch
        assert all(sum(matrix[x][y] for x in kinds) == on['counts'][y] for y in kinds), arch
        rec['transition_matrix'] = matrix
        rec['transition_ids'] = transition_ids
        rec['transitions'] = dict(sorted(trans.items()))
        out['cells'][arch] = rec
    ps = [p for _, _, p in type_tests]
    holm = S.holm(ps)
    out['type_family'] = [{'checkpoint': a, 'type': k, 'mcnemar_p': p, 'bonferroni_p': min(1.0, 4 * p),
                           'holm_p': hp} for (a, k, p), hp in zip(type_tests, holm)]
    # n=1000 pilot: shielded architecture gap
    on_ours = parse_eval('cbf_on.log', 'Sec. VIII n=1000 pilot')
    on_height = parse_eval('cbf_height.log', 'Sec. VIII n=1000 pilot')
    off_ours = parse_eval('cbf_off.log', 'Sec. VIII n=1000 pilot')
    t = S.two_prop(on_ours['succ'], on_ours['n'], on_height['succ'], on_height['n'])
    out['pilot_n1000'] = {'ours_on_pct': 100 * t['pa'], 'height_on_pct': 100 * t['pb'],
                          'ours_off_pct': 100.0 * off_ours['succ'] / off_ours['n'],
                          'shielded_gap_pp': 100 * t['diff'], 'p': t['p']}
    return out


def reciprocity():
    levels = [('02', 0.2), ('04', 0.4), ('06', 0.6), ('08', 0.8), ('10', 1.0)]
    e_dyn = sum(dd / (dd + st) for dd in range(5, 10) for st in range(0, 3)) / 15
    rows, ev = [], {}
    for code, vp in levels:
        o = parse_eval('re_vp%s_ours.log' % code, 'Table VI')
        h = parse_eval('re_vp%s_height.log' % code, 'Table VI')
        ev[code] = (o, h)
        t = tp(o, h)
        rows.append({'visible_prob': vp, 'reactive_pct': 100 * vp * e_dyn, 'ours_pct': t['rate_a'],
                     'height_pct': t['rate_b'], 'gap_pp': t['diff_pp'], 'ci_pp': t['ci_pp'], 'p': t['p'],
                     'survives_bonferroni_5': t['p'] < 0.05 / 5})
    rise = {}
    for idx, arch in ((0, 'ours'), (1, 'height')):
        lo, hi = ev['02'][idx], ev['10'][idx]
        t = tp(hi, lo)
        f0, f1 = 1 - lo['succ'] / N, 1 - hi['succ'] / N
        first = S.two_prop(int(success_vector(hi)[:1000].sum()), 1000, int(success_vector(lo)[:1000].sum()), 1000)
        rise[arch] = {'rise_pp': t['diff_pp'], 'ci_pp': t['ci_pp'], 'p': t['p'],
                      'failure_rate_from_pct': 100 * f0, 'failure_rate_to_pct': 100 * f1,
                      'failure_reduction_pct': 100 * (f1 - f0) / f0,
                      'cohen_h': S.cohen_h(hi['succ'] / N, lo['succ'] / N),
                      'first_1000': {'rise_pp': 100 * first['diff'], 'ci_pp': [100 * first['ci'][0], 100 * first['ci'][1]]}}
    mono = {arch: all(rows[i][k] <= rows[i + 1][k] for i in range(len(rows) - 1))
            for arch, k in (('ours', 'ours_pct'), ('height', 'height_pct'))}
    pilot_o, pilot_h = parse_eval('rp_vp02_ours.log', 'Sec. XIII n=1000 pilot'), parse_eval('rp_vp02_height.log', 'Sec. XIII n=1000 pilot')
    return {'table': rows, 'rise': rise, 'monotone': mono,
            'pilot_n1000_default_gap_pp': 100.0 * (pilot_o['succ'] / pilot_o['n'] - pilot_h['succ'] / pilot_h['n']),
            'reactive_formula': '100 x visible_prob x E[dynamic/(dynamic+static)], dynamic ~ U{5..9}, static ~ U{0,1,2}',
            'design': 'unpaired two-proportion z per level and for the 0.2->1.0 rise; Bonferroni threshold 0.05/5 for the five gaps'}


def taxonomy(primary_dir, repeat_dir):
    sys.path.insert(0, os.path.join(ROOT, 'code'))
    import collision_taxonomy as ct1
    import collision_taxonomy_v2 as ct2
    base = 0.2 * (sum(dd / (dd + st) for dd in range(5, 10) for st in range(0, 3)) / 15)
    out = {'base_rate_reactive': base}
    for label, d in (('primary', primary_dir), ('repeat', repeat_dir)):
        rows = ct2.classify(d)
        n = len(rows)
        nonreact = sum(1 for x in rows if x['reactive'] is False)
        p = nonreact / n
        z = (p - (1 - base)) / math.sqrt(base * (1 - base) / n)
        sec = {k: sum(1 for x in rows if x['sector'] == k) for k in ('side', 'behind', 'front')}
        dirs = {k: sum(1 for x in rows if x['direction'] == k) for k in ('oncoming', 'crossing', 'receding', 'stationary', 'unknown')}
        v1k = {k: sum(1 for x in rows if x['v1_kind'] == k) for k in ('oncoming', 'crossing', 'receding')}
        sp = np.array([x['robot_speed'] for x in rows])
        out[label] = {'n_pedestrian_collisions': n, 'non_reactive': nonreact, 'non_reactive_pct': 100 * p,
                      'wilson_ci_pct': [100 * v for v in S.wilson(nonreact, n)], 'z': z,
                      'p': 2 * S.norm_sf(abs(z)), 'cohen_h': S.cohen_h(p, 1 - base),
                      'bearing_pct': {k: 100.0 * v / n for k, v in sec.items()},
                      'side_or_behind_pct': 100.0 * (sec['side'] + sec['behind']) / n,
                      'approach_direction_counts_v2': dirs,
                      'approach_direction_pct_v2': {k: 100.0 * v / n for k, v in dirs.items()},
                      'relative_motion_counts_v1': v1k,
                      'robot_speed_mean': float(sp.mean()),
                      'reactive_share_of_collisions_pct': 100.0 * (n - nonreact) / n,
                      'involvement_ratio': ((n - nonreact) / n) / base}
        eps = load_dump(d)
        coll = [e for e in eps if str(e['outcome']).startswith('Collision')]
        other = [e for e in coll if not str(e['outcome']).startswith('CollisionHuman')]
        nr_other = 0
        for e in other:
            j, _ = ct1.colliding_human(e['steps'][-1], e['robot_radius'])
            if j is not None and not bool(e['steps'][-1][6][j][5]):
                nr_other += 1
        po = nr_other / len(other) if other else float('nan')
        zo = (po - (1 - base)) / math.sqrt(base * (1 - base) / len(other)) if other else float('nan')
        out[label]['collision_episodes'] = len(coll)
        out[label]['obstacle_or_wall'] = len(other)
        out[label]['nearest_nonreactive_in_obstacle_wall'] = {'k': nr_other, 'n': len(other), 'pct': 100 * po,
                                                               'p': 2 * S.norm_sf(abs(zo))}
        trend = {'fell': 0, 'rose': 0, 'within': 0}
        for e in eps:
            if not str(e['outcome']).startswith('CollisionHuman'):
                continue
            sp_ = [math.hypot(s_[3], s_[4]) for s_ in e['steps']]
            if len(sp_) <= 5:
                trend['within'] += 1
                continue
            dv = sp_[-1] - sp_[-6]
            trend['fell' if dv < -0.005 else ('rose' if dv > 0.005 else 'within')] += 1
        out[label]['speed_trend_0p5s'] = trend
    out['design'] = ('nearest pedestrian by surface gap at the final recorded step; z test of the non-reactive share '
                     'against the population base rate 1 - 0.2 E[dyn/total]; Wilson 95% CI; approach direction from '
                     'code/collision_taxonomy_v2.py (stationary <= 1e-8 m/s); speed trend = last recorded speed minus '
                     'the speed five records earlier, tolerance 0.005 m/s')
    return out


def attribution_and_heading(primary_dir, repeat_dir):
    import taxonomy_sensitivity as ts
    import collision_taxonomy as ct1
    out = {}
    base = ts.BASE_REACT
    for label, d in (('primary', primary_dir), ('repeat', repeat_dir)):
        cols = [e for e in load_dump(d) if str(e['outcome']).startswith('CollisionHuman') and len(e['steps'][0]) >= 7]
        n = len(cols)
        att = {}
        for margin in (0.05, 0.10, 0.20):
            amb = flip = pess = 0
            for e in cols:
                g = ts.gaps_at_impact(e['steps'][-1], e['robot_radius'])
                nearest, runner = g[0], (g[1] if len(g) > 1 else None)
                a = runner is not None and (runner[0] - nearest[0]) <= margin
                amb += a
                flip += bool(a and runner[2] != nearest[2])
                pess += (not nearest[2] and not runner[2]) if a else (not nearest[2])
            pp = pess / n
            z = (pp - (1 - base)) / math.sqrt(base * (1 - base) / n)
            att['%.2f' % margin] = {'ambiguous_pct': 100.0 * amb / n, 'flags_differ_pct': 100.0 * flip / n,
                                    'pessimistic_nonreactive_pct': 100 * pp, 'z': z}
        head = {}
        for deg in (15, 20, 29, 45, 60):
            since = []
            for e in cols:
                g = ts.gaps_at_impact(e['steps'][-1], e['robot_radius'])
                s_ = ct1.steps_since_heading_change(e['steps'], g[0][1], math.radians(deg))
                since.append(None if s_ is None else s_ * e.get('time_step', 0.1))
            w1 = sum(1 for s_ in since if s_ is not None and s_ <= 1.0)
            w2 = sum(1 for s_ in since if s_ is not None and s_ <= 2.0)
            head['%d' % deg] = {'within_1s_pct': 100.0 * w1 / n, 'within_2s_pct': 100.0 * w2 / n}
        out[label] = {'n': n, 'attribution': att, 'heading_change': head}
    out['design'] = ('second pedestrian within the margin of the nearest (surface gap); pessimistic share counts an '
                     'ambiguous case as reactive if either candidate is; heading change = last change above the '
                     'threshold in the pedestrian velocity heading (speed >= 0.05 m/s), within 1 s / 2 s of the last record')
    return out


def responsibility(primary_dir, repeat_dir):
    import responsibility as rs
    out = {}
    for label, d in (('primary', primary_dir), ('repeat', repeat_dir)):
        eps = [e for e in load_dump(d) if e['steps']]
        hc = [e for e in eps if str(e['outcome']).startswith('CollisionHuman')]
        succ = [e for e in eps if str(e['outcome']) == 'ReachGoal']
        r_hc = np.array([x for x in (rs.episode_responsibility(e['steps'], e['robot_radius'])[0] for e in hc) if not np.isnan(x)])
        r_su = np.array([x for x in (rs.episode_responsibility(e['steps'], e['robot_radius'])[0] for e in succ) if not np.isnan(x)])
        lead_first, lead_peak = [], []
        for e in hc:
            first, peak, t_end, _ = rs.evasion_timing(e['steps'], e['robot_radius'])
            if first is not None:
                lead_first.append(t_end - first)
            if peak is not None:
                lead_peak.append(t_end - peak)
        lf, lpk = np.array(lead_first), np.array(lead_peak)
        horizons = {}
        for hz in (3.0, 5.0, 7.5, 10.0):
            v = np.array([x for x in (rs.episode_responsibility(e['steps'], e['robot_radius'], horizon=hz)[0] for e in hc)
                          if not np.isnan(x)])
            horizons['%.1f' % hz] = {'n': int(v.size), 'share_ge_0p5_pct': 100.0 * (v >= 0.5).mean()}
        w = S.welch(r_hc, r_su)
        out[label] = {'human_collisions': len(hc), 'measurable': int(r_hc.size), 'R_robot_mean': float(r_hc.mean()),
                      'R_robot_median': float(np.median(r_hc)), 'share_gt_0p5_pct': 100.0 * (r_hc >= 0.5).mean(),
                      'first_evasive_median_s': float(np.median(lf)), 'peak_evasive_median_s': float(np.median(lpk)),
                      'peak_within_0p5s_pct': 100.0 * (lpk < 0.5).mean(),
                      'horizon_sensitivity': horizons,
                      'control': {'successes_measurable': int(r_su.size), 'success_mean': float(r_su.mean()),
                                  'diff': w['diff'], 'ci': list(w['ci']), 'p_exact_welch': w['p'], 'df': w['df'],
                                  'p_normal_approx_as_printed_by_responsibility_py': 2 * S.norm_sf(abs(w['t']))}}
        if label == 'primary':
            eps_with, samples = 0, 0
            for e in hc:
                credited = 0
                st = e['steps']
                for i in range(1, len(st)):
                    p_, c_ = st[i - 1], st[i]
                    dt = c_[0] - p_[0]
                    if dt <= 0 or len(c_[-1]) != len(p_[-1]):
                        continue
                    if None in (c_[1], c_[2], c_[3], c_[4], p_[3], p_[4]):
                        continue
                    for h, hp in zip(c_[-1], p_[-1]):
                        if None in (h[0], h[1], h[2], h[3], hp[2], hp[3]) or bool(h[5]):
                            continue
                        now = rs.conflict((c_[1], c_[2]), (c_[3], c_[4]), e['robot_radius'], (h[0], h[1]), (h[2], h[3]), h[4])
                        nc = rs.conflict((c_[1], c_[2]), (c_[3], c_[4]), e['robot_radius'], (h[0], h[1]), (hp[2], hp[3]), h[4])
                        if -(now - nc) * dt > 1e-12:
                            credited += 1
                samples += credited
                eps_with += credited > 0
            out[label]['nonreactive_credit'] = {'episodes': eps_with, 'of': len(hc), 'agent_step_samples': samples}
    out['design'] = ('agent-only negative-contribution share (code/responsibility.py, horizon 5 s) over pedestrian '
                     'collisions with measurable conflict; control vs every eighth success by Welch t with exact t tail '
                     '(responsibility.py itself prints a normal approximation)')
    return out


def ics(primary_manifest):
    out = {'design': ('closed-prefix search [t0, t0+k] of code/ics_analysis_v2.py; historical cohort = collision '
                      'records with >= 25 states and a non-reactive nearest pedestrian (385); no-escape = '
                      'initial_contact + no_surviving_sequence'), 'historical_385': {}, 'per_horizon_386': {}}
    for pad in ('0.0', '0.049', '0.1'):
        f = os.path.join(ROOT, 'results', 'v2', 'ics_v2_pad%s.json' % pad)
        d = json.load(open(use(f, 'Table IV')))
        assert d['n_files'] == primary_manifest['n_files'] and d['traj_dir_sha256'] == primary_manifest['traj_dir_sha256'], (f, 'ICS trajectory provenance')
        rec = {}
        for k in ('5', '10', '20'):
            c = d['counts'][k]['nonreact']
            n = sum(c.values())
            fail = c['initial_contact'] + c['no_surviving_sequence']
            rec[k] = {'no_escape': fail, 'n': n, 'pct': 100.0 * fail / n, 'initial_contact': c['initial_contact'],
                      'aborted': c['aborted'], 'escape_pct': 100.0 * (n - fail) / n}
        comp = {'human': 0, 'obstacle': 0, 'wall': 0}
        for ep in d['episodes']:
            if ep['bucket'] == 'nonreact':
                o = ep['outcome']
                comp['human' if o.startswith('CollisionHuman') else ('obstacle' if o.startswith('CollisionObs') else 'wall')] += 1
        out['historical_385'][pad] = {'rows': rec, 'excluded': d['excluded'], 'composition': comp,
                                      'witness_replay_failures': d['witness_replay_failures'],
                                      'selfcheck': d['selfcheck'], 'episodes_searched': d['n_episodes']}
        f2 = os.path.join(ROOT, 'results', 'v2', 'ics_v2_pad%s_perhorizon.json' % pad)
        if os.path.exists(f2):
            d2 = json.load(open(use(f2, 'Sec. VII separate 386-case cohort')))
            assert d2['n_files'] == primary_manifest['n_files'] and d2['traj_dir_sha256'] == primary_manifest['traj_dir_sha256'], (f2, 'ICS trajectory provenance')
            out['per_horizon_386'][pad] = {
                k: {'no_escape': d2['counts'][k]['nonreact']['initial_contact'] + d2['counts'][k]['nonreact']['no_surviving_sequence'],
                    'n': sum(d2['counts'][k]['nonreact'].values())} for k in ('5', '10', '20')}
    use(os.path.join(ROOT, 'results', 'logs', 'ics_full_results.json'), 'Table IV v1 (historical, comparison only)')
    v1 = json.load(open(os.path.join(ROOT, 'results', 'logs', 'ics_full_results.json')))
    out['v1_historical_no_escape'] = {pad: {k: v1[pad][k]['nonreact'][1] for k in ('5', '10', '20')} for pad in v1}
    return out


def social():
    released = parse_eval('re_vp02_height.log', 'Table VIII')
    ft = parse_eval('eval_exp_heightft.log', 'Table VIII')
    ours = parse_eval('re_vp02_ours.log', 'Table VIII')
    ih = intrusion('intrusion_test_re_vp02_height.json', 'Table VIII')
    ift = intrusion('intrusion_test_exp_heightft.json', 'Table VIII')
    io = intrusion('intrusion_test_re_vp02_ours.json', 'Table VIII')
    w_or = S.welch(io, ih)
    w_of = S.welch(io, ift)
    w_fr = S.welch(ift, ih)
    pt = S.paired_t(ift, ih)
    mw = S.mann_whitney(ih, io)
    nav = {k: e['nav_ep'] for k, e in (('released', released), ('ft', ft), ('ours', ours))}
    s2 = parse_eval('eval_exp_nospeak_s2.log', 'Sec. XI other oracle seeds')['intrusion_summary']
    s3 = parse_eval('eval_exp_nospeak_s3.log', 'Sec. XI other oracle seeds')['intrusion_summary']
    gap = ih.mean() - io.mean()
    closed = ih.mean() - ift.mean()
    return {
        'success_pct': {'released': pct(released['succ']), 'ft350': pct(ft['succ']), 'ours': pct(ours['succ'])},
        'ours_vs_ft_success_p': tp(ours, ft)['p'],
        'ft_vs_released_success': tp(ft, released),
        'ft_vs_released_success_mcnemar_p': S.mcnemar_exact(
            int(((success_vector(ft) == 1) & (success_vector(released) == 0)).sum()),
            int(((success_vector(ft) == 0) & (success_vector(released) == 1)).sum())),
        'nav_time_s': {k: float(v.mean()) for k, v in nav.items()},
        'nav_p_ours_vs_released': S.welch(nav['ours'], nav['released'])['p'],
        'nav_p_ft_vs_released': S.welch(nav['ft'], nav['released'])['p'],
        'intrusion_mean': {'released': float(ih.mean()), 'ft350': float(ift.mean()), 'ours': float(io.mean())},
        'intrusion_sd': {'released': float(ih.std(ddof=1)), 'ft350': float(ift.std(ddof=1)), 'ours': float(io.std(ddof=1))},
        'intrusion_median': {'released': float(np.median(ih)), 'ft350': float(np.median(ift)), 'ours': float(np.median(io))},
        'zero_intrusion_pct': {'released': 100 * float((ih == 0).mean()), 'ft350': 100 * float((ift == 0).mean()),
                               'ours': 100 * float((io == 0).mean())},
        'relative_pct': {'ft_vs_released': 100 * (ift.mean() - ih.mean()) / ih.mean(),
                         'ours_vs_released': 100 * (io.mean() - ih.mean()) / ih.mean(),
                         'ours_vs_ft': 100 * (io.mean() - ift.mean()) / ift.mean()},
        'cohen_d': {'ours_vs_released': S.cohen_d(io, ih), 'ours_vs_ft': S.cohen_d(io, ift)},
        'welch_ours_vs_released': {'t': w_or['t'], 'p': w_or['p']},
        'welch_ours_vs_ft': {'t': w_of['t'], 'p': w_of['p'], 'ci': list(w_of['ci'])},
        'continuation_vs_released': {'unpaired_p': w_fr['p'], 'paired_p': pt['p'], 'paired_t': pt['t']},
        'mann_whitney_released_vs_ours': mw,
        'gap_points': gap, 'closed_by_ft_points': closed, 'closed_share_pct': 100 * closed / gap,
        'remaining_points': gap - closed,
        'other_oracle_seeds_summary_intrusion': {'seed2': s2, 'seed3': s3,
                                                 'below_ft_pct': [100 * (ift.mean() - s2) / ift.mean(),
                                                                  100 * (ift.mean() - s3) / ift.mean()]},
        'design': ('ours = seed 1 with ground-truth futures; intrusion = per-episode share of steps inside the 0.25 m '
                   'discomfort zone; Welch t (exact tail) on per-episode arrays; HEIGHT vs its continuation also as a '
                   'paired t over episode index (same seed sequence; scene identity not archived); navigation time = '
                   '(steps - 1) x 0.1 s over successful episodes; seeds 2 and 3 have only the logged two-decimal means')}


def deployable():
    released = parse_eval('re_vp02_height.log', 'Table IX')
    ft = parse_eval('eval_exp_heightft.log', 'Table IX')
    ih = intrusion('intrusion_test_re_vp02_height.json', 'Table IX')
    ift = intrusion('intrusion_test_exp_heightft.json', 'Table IX')
    io = intrusion('intrusion_test_re_vp02_ours.json', 'Table IX')
    rows = []
    for s in (1, 2, 3):
        ev = parse_eval('eval_exp_nopredict_s%d.log' % s, 'Table IX')
        a = intrusion('intrusion_test_exp_nopredict_s%d.json' % s, 'Table IX')
        rows.append({'seed': s, 'success_pct': pct(ev['succ']),
                     'vs_ft': tp(ev, ft), 'vs_released': tp(ev, released),
                     'intrusion_mean': float(a.mean()),
                     'vs_ft_intr': {'rel_pct': 100 * (a.mean() - ift.mean()) / ift.mean(), 'p': S.welch(a, ift)['p']},
                     'vs_released_intr': {'rel_pct': 100 * (a.mean() - ih.mean()) / ih.mean(), 'p': S.welch(a, ih)['p']},
                     'vs_oracle_seed1_intr_p': S.welch(a, io)['p'],
                     'nav_time_s': float(ev['nav_ep'].mean())})
    rates = np.array([x['success_pct'] for x in rows])
    return {'rows': rows, 'mean_pct': float(rates.mean()), 'sd_pp': float(rates.std(ddof=1)),
            'design': 'each seed tested separately (unpaired) against HEIGHT-350M and the released checkpoint'}


def acoustic():
    a = parse_eval('h1_exp_full_s1_n3500.log', 'Sec. XIV excluded experiment')
    b = parse_eval('eval_exp_nospeak_s1.log', 'Sec. XIV excluded experiment')
    t = tp(a, b)
    return {'with_beep': a['succ'], 'without': b['succ'], 'diff_pp': t['diff_pp'], 'p': t['p']}


def path_jump(primary_dir):
    lengths = np.array(json.load(open(log('intrusion_test_exp_nospeak_s1.json', 'Sec. XI path-length artefact')))['path_lengths'])
    eps = {e['episode']: e for e in load_dump(primary_dir)}
    jumps, resid = [], []
    for k, e in eps.items():
        nxt = eps.get(k + 1)
        if nxt is None:
            continue
        st = e['steps']
        rec = sum(math.hypot(st[i][1] - st[i - 1][1], st[i][2] - st[i - 1][2]) for i in range(1, len(st)))
        jump = math.hypot(nxt['steps'][0][1] - st[-1][1], nxt['steps'][0][2] - st[-1][2])
        jumps.append(jump)
        resid.append(lengths[k] - (rec + jump))
    return {'consecutive_pairs': len(jumps), 'mean_jump_m': float(np.mean(jumps)),
            'max_abs_residual_m': float(np.max(np.abs(resid))), 'logged_mean_path_m': float(lengths.mean()),
            'design': 'for consecutively numbered dumped episodes, logged path length = recorded path + jump to the next start'}


# ----------------------------------------------------------------------------- output
def md_table(title, headers, rows):
    out = ['### ' + title, '', '| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out) + '\n'


def fmt_p(p):
    return '%.3g' % p


def write_md(res, path):
    L = ['# Paper statistics, recomputed (%s)' % VERSION, '',
         'Generated by stats/paper3_stats.py from the archived inputs listed at the end, with their SHA-256.', '']
    t = res['reproduction']
    L.append(md_table('Table I: reproduction', ['method', 'reproduced %', 'published %', 'delta pp'],
                      [[x['method'], '%.2f' % x['reproduced_pct'], x['published_pct'], '%+.2f' % x['delta_pp']] for x in t['table']]))
    o = res['oracle_range']
    L.append(md_table('Table II: oracle vs sensing range', ['range m', 'occluded %', 'oracle %', 'gap pp', '95% CI', 'p'],
                      [[x['range_m'], '%.2f' % x['occluded_pct'], '%.2f' % x['oracle_pct'], '%+.2f' % x['gap_pp'],
                        '[%+.2f, %+.2f]' % tuple(x['ci_pp']), fmt_p(x['p'])] for x in o['table']]))
    a = res['architecture']
    L.append(md_table('Table III: architecture seeds', ['quantity', 'value'],
                      [['seeds %', ', '.join('%.2f' % v for v in a['seed_pct'])],
                       ['mean +/- SD', '%.2f +/- %.2f' % (a['mean_pct'], a['sd_pp'])],
                       ['95% CI on mean (df 2)', '[%.1f, %.1f]' % tuple(a['ci_mean_pct'])],
                       ['diff vs HEIGHT', '%+.2f pp, t=%.2f' % (a['diff_pp'], a['t'])],
                       ['TOST 90% CI / p', '[%+.2f, %+.2f] / %.3f' % (a['tost']['ci90_pp'][0], a['tost']['ci90_pp'][1], a['tost']['p'])]]))
    rows = []
    for k, c in res['shield']['cells'].items():
        for typ in ('success', 'human', 'obstacle', 'wall', 'timeout'):
            pp = c['paired'][typ]
            rows.append([k, typ, '%d -> %d' % (pp['off'], pp['on']), '%d / %d' % (pp['off_only'], pp['on_only']),
                         fmt_p(pp['mcnemar_p']), fmt_p(pp['unpaired_p'])])
    L.append(md_table('Table V: shield (pedestrian-only), paired by episode index', ['checkpoint', 'outcome', 'off -> on',
                      'off-only / on-only', 'exact McNemar p', 'unpaired p'], rows))
    L.append(md_table('Shield: exploratory family of four', ['checkpoint', 'type', 'p', 'Bonferroni', 'Holm'],
                      [[x['checkpoint'], x['type'], fmt_p(x['mcnemar_p']), fmt_p(x['bonferroni_p']), fmt_p(x['holm_p'])]
                       for x in res['shield']['type_family']]))
    rc = res['reciprocity']
    L.append(md_table('Table VI: reciprocity', ['visible_prob', 'reactive %', 'ours %', 'HEIGHT %', 'gap pp', 'p'],
                      [[x['visible_prob'], '%.1f' % x['reactive_pct'], '%.2f' % x['ours_pct'], '%.2f' % x['height_pct'],
                        '%+.2f' % x['gap_pp'], fmt_p(x['p'])] for x in rc['table']]))
    tx = res['taxonomy']
    L.append(md_table('Table VII: taxonomy', ['quantity', 'primary', 'repeat'], [
        ['non-reactive', '%d/%d (%.1f%%)' % (tx['primary']['non_reactive'], tx['primary']['n_pedestrian_collisions'], tx['primary']['non_reactive_pct']),
         '%d/%d' % (tx['repeat']['non_reactive'], tx['repeat']['n_pedestrian_collisions'])],
        ['approach direction (v2)', ' / '.join('%s %.1f%%' % (k, v) for k, v in tx['primary']['approach_direction_pct_v2'].items() if k != 'unknown'),
         ' / '.join('%s %.1f%%' % (k, v) for k, v in tx['repeat']['approach_direction_pct_v2'].items() if k != 'unknown')],
        ['speed trend 0.5 s (fell/rose/within)', '%(fell)d/%(rose)d/%(within)d' % tx['primary']['speed_trend_0p5s'],
         '%(fell)d/%(rose)d/%(within)d' % tx['repeat']['speed_trend_0p5s']]]))
    ic = res['ics']['historical_385']
    L.append(md_table('Table IV: no-escape fraction, closed prefix, 385 cohort', ['look-back', 'pad 0', 'pad 0.049', 'pad 0.10'],
                      [['%.1f s' % (int(k) / 10), *['%d/%d (%.1f%%)' % (ic[p]['rows'][k]['no_escape'], ic[p]['rows'][k]['n'], ic[p]['rows'][k]['pct'])
                                                    for p in ('0.0', '0.049', '0.1')]] for k in ('5', '10', '20')]))
    so = res['social']
    L.append(md_table('Table VIII: social metrics', ['quantity', 'value'], [
        ['intrusion mean released / 350M / ours', '%.4f / %.4f / %.4f' % tuple(so['intrusion_mean'].values())],
        ['ours vs released Welch', 't=%.2f p=%.3g' % (so['welch_ours_vs_released']['t'], so['welch_ours_vs_released']['p'])],
        ['ours vs 350M Welch', 't=%.2f p=%.3g CI [%.2f, %.2f]' % (so['welch_ours_vs_ft']['t'], so['welch_ours_vs_ft']['p'], *so['welch_ours_vs_ft']['ci'])],
        ['continuation vs released (paired / unpaired)', '%.4g / %.4g' % (so['continuation_vs_released']['paired_p'], so['continuation_vs_released']['unpaired_p'])],
        ['nav time released / 350M / ours', '%.2f / %.2f / %.2f' % tuple(so['nav_time_s'].values())]]))
    dp = res['deployable']
    L.append(md_table('Table IX: deployable', ['seed', 'success %', 'p vs 350M', 'intrusion', 'rel/p vs 350M', 'rel/p vs released', 'p vs oracle s1'],
                      [[x['seed'], '%.2f' % x['success_pct'], fmt_p(x['vs_ft']['p']), '%.2f' % x['intrusion_mean'],
                        '%.1f%% / %.3g' % (x['vs_ft_intr']['rel_pct'], x['vs_ft_intr']['p']),
                        '%.1f%% / %.3g' % (x['vs_released_intr']['rel_pct'], x['vs_released_intr']['p']),
                        fmt_p(x['vs_oracle_seed1_intr_p'])] for x in dp['rows']]))
    L.append('## Inputs\n')
    L.append(md_table('SHA-256 of every input read', ['file', 'sha256', 'used for'],
                      [[k, v['sha256'], '; '.join(v['roles'])] for k, v in sorted(res['inputs'].items())]))
    with open(path, 'w') as f:
        f.write('\n'.join(L))


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default='.')
    ap.add_argument('--primary')
    ap.add_argument('--repeat')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ROOT = os.path.abspath(a.root)
    INPUTS.clear()
    out_dir = a.out or os.path.join(ROOT, 'stats', 'outputs', OUTPUT_VERSION)
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, 'code'))
    for code_file in ('collision_taxonomy.py', 'collision_taxonomy_v2.py', 'taxonomy_sensitivity.py', 'responsibility.py'):
        use(os.path.join(ROOT, 'code', code_file), 'analysis code imported by the driver')
    for code_file in ('paper3_stats.py', 'statlib.py', 'provenance.py'):
        use(os.path.join(ROOT, 'stats', code_file), 'released statistical procedure')
    use(os.path.join(ROOT, 'code', 'ics_analysis_v2.py'), 'ICS output-generating procedure (separately run)')
    res = {'version': VERSION, 'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__},
           'trajectory_provenance': {}}
    with tempfile.TemporaryDirectory(prefix='paper3_stats_') as tmp:
        dirs = []
        for label, tarname, supplied in (('primary', 'trajectories_n3500.tar.gz', a.primary),
                                         ('repeat', 'trajectories_n3500_repeat.tar.gz', a.repeat)):
            archive = use(os.path.join(ROOT, 'data', tarname), 'trajectory dump (payload bytes verified)')
            directory, manifest = prepare_trajectories(archive, tmp, supplied)
            dirs.append(directory)
            res['trajectory_provenance'][label] = {'archive': os.path.relpath(archive, ROOT), **manifest}
        primary, repeat = dirs
        res['reproduction'] = reproduction()
        res['protocol'] = protocol()
        res['oracle_range'] = oracle_range()
        res['architecture'] = architecture()
        res['shield'] = shield()
        res['reciprocity'] = reciprocity()
        res['taxonomy'] = taxonomy(primary, repeat)
        res['attribution_heading'] = attribution_and_heading(primary, repeat)
        res['responsibility'] = responsibility(primary, repeat)
        res['ics'] = ics(res['trajectory_provenance']['primary'])
        res['social'] = social()
        res['deployable'] = deployable()
        res['acoustic'] = acoustic()
        res['path_jump'] = path_jump(primary)
    res['inputs'] = INPUTS
    with open(os.path.join(out_dir, 'paper3_stats.json'), 'w') as f:
        json.dump(res, f, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    write_md(res, os.path.join(out_dir, 'paper3_stats.md'))
    print('%s: %d inputs hashed; written to %s' % (VERSION, len(INPUTS), out_dir))


if __name__ == '__main__':
    main()
