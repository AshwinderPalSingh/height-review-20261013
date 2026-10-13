"""Compare every verdict of the frozen v1 search with ics_analysis_v2 on the same records.

Expectation: v2 differs from v1 only where the start state is already in contact. There,
v2 reports initial_contact; v1 reports whatever the open-start search found, which v2 also
records as `open_start`. Every other (episode, horizon) must agree exactly.

    python3 tests/compare_ics_v1_v2.py v1_pad0.049.json ics_v2_pad0.049.json [...pairs]
"""
import json
import sys


def compare(v1_path, v2_path):
    v1 = json.load(open(v1_path))
    v2 = json.load(open(v2_path))
    assert abs(v1['pad'] - v2['pad']) < 1e-12, 'pad mismatch'
    v1rows = {r['episode']: r for r in v1['rows']}
    mismatches, flips, contact_agree = [], [], 0
    n = 0
    for ep in v2['episodes']:
        r1 = v1rows.get(ep['episode'])
        assert r1 is not None, 'episode %s missing from v1' % ep['episode']
        assert r1['bucket'] == ep['bucket']
        for k, rec in ep['horizons'].items():
            n += 1
            old = r1['verdicts'][k]
            if rec['status'] == 'initial_contact':
                if (rec['open_start'] == 'survives_horizon') != old:
                    mismatches.append((ep['episode'], k, 'open_start', rec['open_start'], old))
                elif old:
                    flips.append({'episode': ep['episode'], 'horizon_steps': int(k), 'bucket': ep['bucket'],
                                  'cause': rec['initial_contact']})
                else:
                    contact_agree += 1
            elif (rec['status'] == 'survives_horizon') != old:
                mismatches.append((ep['episode'], k, rec['status'], old))
    return {'pad': v2['pad'], 'pairs_compared': n, 'mismatches': mismatches, 'flips': flips,
            'initial_contact_already_no_escape_in_v1': contact_agree}


if __name__ == '__main__':
    args = sys.argv[1:]
    out = [compare(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    print(json.dumps(out, indent=1))
    if any(o['mismatches'] for o in out):
        sys.exit(1)
