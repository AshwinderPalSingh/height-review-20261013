#!/usr/bin/env python3
"""Every number in the Paper 1 lab report's acoustic and generalization sections, from the primary logs.

    python3 -I -B compute_stats.py      # reads logs/, writes stats.json

The logs are byte copies of gdrive:HEIGHT_Backup/logs/<name> (downloaded 22 Oct 2026). Outcomes are
tallied from the per-episode lines; navigation time, intrusion ratio and the acoustic counters are
read from each log's closing summary. Success contrasts use the pooled two-proportion z test with a
Wald 95% interval, as in the rest of Paper 1; navigation time uses Welch's t from the logged
mean, SD and n (successful episodes only, so the episode subsets differ between runs).
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / 'results' / 'logs'
RUNS = {
    'height': 'eval_heightpub.log', 'nospeak': 'eval_exp_nospeak_s1.log',
    'noreturn': 'h1_exp_noreturn_s1_n3500.log', 'full': 'h1_exp_full_s1_n3500.log',
    'nocal': 'eval_nocal_s1_n3500.log',
    'full_scenes2': 'evalseed2_exp_full_s1.log', 'full_scenes3': 'evalseed3_exp_full_s1.log',
    'nospeak_scenes2': 'evalseed2_exp_nospeak_s1.log', 'nospeak_scenes3': 'evalseed3_exp_nospeak_s1.log',
    'humans12_height': 'gen_more_human_heightpub.log', 'humans12_nospeak': 'gen_more_human_nospeak.log',
    'humans12_noreturn': 'gen_more_human_noreturn.log', 'humans12_full': 'gen_more_human_full.log',
    'obstacles15_height': 'gen_more_obs_heightpub.log', 'obstacles15_nospeak': 'gen_more_obs_nospeak.log',
    'obstacles15_noreturn': 'gen_more_obs_noreturn.log',
}
KEYS = {'Success': 'success', 'Collision with human': 'human', 'Collision with obstacle': 'obstacle',
        'Collision with wall': 'wall', 'Time out': 'timeout'}


def parse(path):
    lines = path.read_text(errors='replace').splitlines()
    out = collections.Counter()
    for i, line in enumerate(lines):
        if re.match(r'Episode \d+ ends in \d+', line):
            j = i + 1
            while not lines[j].strip():
                j += 1
            out[KEYS[lines[j].strip()]] += 1
    text = '\n'.join(lines)
    nav = re.findall(r'nav time \(successes\): ([\d.]+) \+/- ([\d.]+) over n=(\d+)', text)[-1]
    beeps = re.findall(r'beeps/episode: ([\d.]+) \+/- ([\d.]+) \| beeps per 100 steps: ([\d.]+)', text)[-1]
    heard = re.findall(r'localized-by-audio per episode: ([\d.]+) \+/- ([\d.]+)', text)[-1]
    yield_ = re.findall(r'information yield \(occluded humans localized per beep\): (\S+)', text)[-1]
    intrusion = re.findall(r'average intrusion ratio: ([\d.]+)', text)[-1]
    n = sum(out.values())
    assert n == 3500 and out['success'] == int(nav[2])
    return {'episodes': n, **{k: out[k] for k in KEYS.values()},
            'success_pct': 100 * out['success'] / n,
            'nav_time_mean': float(nav[0]), 'nav_time_sd': float(nav[1]),
            'beeps_per_episode': float(beeps[0]), 'beeps_per_100_steps': float(beeps[2]),
            'occluded_heard_per_episode': float(heard[0]), 'yield_per_beep': yield_,
            'intrusion_ratio': float(intrusion), 'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def two_prop(a, b, n=3500):
    p1, p2 = a / n, b / n
    pool = (a + b) / (2 * n)
    z = (p1 - p2) / math.sqrt(pool * (1 - pool) * 2 / n)
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    d = 100 * (p1 - p2)
    return {'diff_pp': round(d, 2), 'ci95_pp': [round(d - 196 * se, 2), round(d + 196 * se, 2)],
            'z': round(z, 3), 'p': float('%.3g' % math.erfc(abs(z) / math.sqrt(2)))}


def welch(r1, r2):
    m1, s1, n1 = r1['nav_time_mean'], r1['nav_time_sd'], r1['success']
    m2, s2, n2 = r2['nav_time_mean'], r2['nav_time_sd'], r2['success']
    t = (m1 - m2) / math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    return {'diff_s': round(m1 - m2, 2), 't': round(t, 2), 'p_normal_approx': float('%.3g' % math.erfc(abs(t) / math.sqrt(2)))}


R = {k: parse(LOGS / v) for k, v in RUNS.items()}
S = lambda k: R[k]['success']
contrasts = {
    'nospeak_vs_height': two_prop(S('nospeak'), S('height')),
    'noreturn_vs_nospeak': two_prop(S('noreturn'), S('nospeak')),
    'full_vs_nospeak': two_prop(S('full'), S('nospeak')),
    'full_vs_noreturn': two_prop(S('full'), S('noreturn')),
    'nocal_vs_full': two_prop(S('nocal'), S('full')),
    'full_vs_height': two_prop(S('full'), S('height')),
    'ped_collision_full_vs_nospeak': two_prop(R['full']['human'], R['nospeak']['human']),
    'humans12_nospeak_vs_height': two_prop(S('humans12_nospeak'), S('humans12_height')),
    'humans12_noreturn_vs_height': two_prop(S('humans12_noreturn'), S('humans12_height')),
    'humans12_noreturn_vs_nospeak': two_prop(S('humans12_noreturn'), S('humans12_nospeak')),
    'humans12_full_vs_noreturn': two_prop(S('humans12_full'), S('humans12_noreturn')),
    'humans12_ped_collision_full_vs_noreturn': two_prop(R['humans12_full']['human'], R['humans12_noreturn']['human']),
    'obstacles15_nospeak_vs_height': two_prop(S('obstacles15_nospeak'), S('obstacles15_height')),
    'obstacles15_noreturn_vs_height': two_prop(S('obstacles15_noreturn'), S('obstacles15_height')),
    'obstacles15_noreturn_vs_nospeak': two_prop(S('obstacles15_noreturn'), S('obstacles15_nospeak')),
}
scene_sets = [('full', 'nospeak'), ('full_scenes2', 'nospeak_scenes2'), ('full_scenes3', 'nospeak_scenes3')]
replication = {'per_scene_set_pp': [round(100 * (S(a) - S(b)) / 3500, 2) for a, b in scene_sets],
               'pooled': two_prop(sum(S(a) for a, _ in scene_sets), sum(S(b) for _, b in scene_sets), 3 * 3500),
               'note': 'same two checkpoints on three test-scene sets; not independent training seeds'}
nav = {'full_vs_nospeak': welch(R['full'], R['nospeak']), 'noreturn_vs_nospeak': welch(R['noreturn'], R['nospeak'])}
out = {'runs': R, 'success_contrasts': contrasts, 'scene_set_replication': replication, 'nav_time': nav}
(HERE / 'stats.json').write_text(json.dumps(out, indent=1) + '\n')
for k, v in R.items():
    print('%-22s %4d %6.2f%%  ped %3d obs %3d wall %d to %2d  nav %.2f  beeps %.2f  heard %.2f  intr %.2f'
          % (k, v['success'], v['success_pct'], v['human'], v['obstacle'], v['wall'], v['timeout'],
             v['nav_time_mean'], v['beeps_per_episode'], v['occluded_heard_per_episode'], v['intrusion_ratio']))
for k, v in contrasts.items():
    print('%-40s %s' % (k, v))
print('replication', replication)
print('nav', nav)
