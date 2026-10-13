"""Re-run the frozen v1 search (code/ics_analysis.py, unmodified) and record every verdict.

Used only to validate ics_analysis_v2.py: the two must agree everywhere except where the
start state is already in contact. v1 reads its padding from the module global PAD, so the
global is set before any call; nothing else in v1 is touched.

    python3 tests/ics_v1_rerun.py <trajectory_dir> --pad 0.049 --out v1_pad0.049.json
"""
import argparse
import glob
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'code'))
import ics_analysis as v1  # noqa: E402

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traj_dir')
    ap.add_argument('--pad', type=float, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    v1.PAD = a.pad
    rows = []
    counts = {k: {'nonreact': [0, 0], 'react': [0, 0]} for k in (5, 10, 20)}
    for fn in sorted(glob.glob(os.path.join(a.traj_dir, 'ep_*.pkl'))):
        with open(fn, 'rb') as f:
            d = pickle.load(f)
        if not str(d['outcome']).startswith('Collision') or len(d.get('steps', [])) < 25:
            continue
        if len(d['steps'][0]) < 7:
            continue
        steps = d['steps']
        r_rob = d['robot_radius']
        final_humans = [h for h in steps[-1][6] if h[0] is not None]
        rx, ry = steps[-1][1], steps[-1][2]
        if final_humans:
            j = int(np.argmin([np.hypot(rx - h[0], ry - h[1]) - (r_rob + h[4]) for h in final_humans]))
            bucket = 'react' if bool(final_humans[j][5]) else 'nonreact'
        else:
            bucket = 'nonreact'
        verdicts = {}
        for k in (5, 10, 20):
            t0 = len(steps) - 1 - k
            if t0 < 0:
                continue
            ok = v1.escapable(steps, t0, k, r_rob, d.get('robot_v_max', 0.5),
                              d.get('time_step', 0.1), d.get('obstacle_vertices'))
            verdicts[str(k)] = bool(ok)
            counts[k][bucket][0 if ok else 1] += 1
        rows.append({'episode': d.get('episode'), 'bucket': bucket, 'verdicts': verdicts})
    with open(a.out, 'w') as f:
        json.dump({'pad': a.pad, 'v1_sha256_note': 'code/ics_analysis.py as shipped at ac60409',
                   'counts': {str(k): v for k, v in counts.items()}, 'rows': rows}, f, indent=1)
    for k in (5, 10, 20):
        print('pad %.3f  %.1f s  non-reactive [escapable, inevitable] = %s  reactive = %s'
              % (a.pad, k * 0.1, counts[k]['nonreact'], counts[k]['react']))


if __name__ == '__main__':
    main()
