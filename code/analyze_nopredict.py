"""Self-interpreting analysis for the predict_method='none' retrain (21 Sep run).

Reads the three seed evaluation logs and per-episode intrusion JSONs, runs every test
the paper needs, and prints a plain-language VERDICT with the exact LaTeX to paste into
each \\pending block of papers/p3_journal/paper3.tex.

Designed to be run by someone with NO statistics background and NO assistant available.
Everything it prints is meant to be copied, not interpreted.

Usage:  python3 analyze_nopredict.py            (from /workspace/HEIGHT)
        python3 analyze_nopredict.py --logs logs  --seeds 1 2 3
Re-run any time; it only reads files.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

# ------------------------------------------------------------------ reference values
# HEIGHT's released checkpoint, evaluated in our codebase, n=3500, matched config.
HEIGHT_SUCC = 3081
HEIGHT_N = 3500
HEIGHT_INTR_MEAN = 5.6578
HEIGHT_INTR_SD = 7.0961
HEIGHT_PATH_MEAN = 10.32
HEIGHT_PATH_SD = 2.55

# Our ORACLE system (predict_method='truth'), for the "how much did privileged info buy" row.
ORACLE_SUCC_MEAN = 88.08
ORACLE_SUCC_SD = 0.49
ORACLE_INTR_MEAN = 4.7471
ORACLE_INTR_SD = 6.1380

# HEIGHT fine-tuned to 350M (Experiments/height_ft150.sh). Read from its logs if present;
# this is the COMPUTE-MATCHED control for the 350M nopredict arm. Released HEIGHT is
# 199.8M; our oracle system received ~426M (audit 17 Sep), so neither of those is matched.
HEIGHTFT_LOG = 'eval_exp_heightft.log'
HEIGHTFT_JSON = 'intrusion_test_exp_heightft.json'


def read_heightft(logdir):
    log = os.path.join(logdir, HEIGHTFT_LOG)
    js = os.path.join(logdir, HEIGHTFT_JSON)
    out = {'ok': False}
    if not os.path.exists(log):
        return out
    txt = open(log).read()
    m = re.search(r'nav time \(successes\):.*?over n=(\d+)', txt)
    if not m:
        return out
    out['succ'] = int(m.group(1))
    m = re.search(r'intrusion ratio \(per-episode\): ([\d.]+) \+/- ([\d.]+)', txt)
    if m:
        out['intr_mean'], out['intr_sd'] = float(m.group(1)), float(m.group(2))
    if os.path.exists(js):
        out['intr_ep'] = json.load(open(js)).get('too_close_ratios', [])
    out['ok'] = True
    return out


def two_prop(a, na, b, nb):
    p1, p2 = a / na, b / nb
    p = (a + b) / (na + nb)
    se = math.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    z = (p1 - p2) / se if se else 0.0
    pv = math.erfc(abs(z) / math.sqrt(2))
    sed = math.sqrt(p1 * (1 - p1) / na + p2 * (1 - p2) / nb)
    d = (p1 - p2) * 100
    return d, d - 1.96 * sed * 100, d + 1.96 * sed * 100, z, pv


def welch(m1, s1, n1, m2, s2, n2):
    v = s1 * s1 / n1 + s2 * s2 / n2
    se = math.sqrt(v)
    t = (m1 - m2) / se if se else 0.0
    pv = math.erfc(abs(t) / math.sqrt(2))
    sp = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp if sp else 0.0
    return m1 - m2, (m1 - m2) - 1.96 * se, (m1 - m2) + 1.96 * se, t, pv, d


def ltx_p(pv):
    if pv >= 0.001:
        return '%.3f' % pv
    e = int(math.floor(math.log10(pv))); m = pv / 10 ** e
    return '%.1f\\times10^{%d}' % (m, e)


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def read_seed(logdir, s):
    log = os.path.join(logdir, 'eval_exp_nopredict_s%d.log' % s)
    js = os.path.join(logdir, 'intrusion_test_exp_nopredict_s%d.json' % s)
    out = {'seed': s, 'log': log, 'json': js, 'ok': False}
    if not os.path.exists(log):
        out['error'] = 'log missing'
        return out
    txt = open(log).read()
    m = re.search(r'nav time \(successes\):.*?over n=(\d+)', txt)
    if not m:
        out['error'] = 'log has no final summary (run incomplete?)'
        return out
    out['succ'] = int(m.group(1))
    m = re.search(r'intrusion ratio \(per-episode\): ([\d.]+) \+/- ([\d.]+)', txt)
    if m:
        out['intr_mean'], out['intr_sd'] = float(m.group(1)), float(m.group(2))
    m = re.search(r'path length: ([\d.]+) \+/- ([\d.]+)', txt)
    if m:
        out['path_mean'], out['path_sd'] = float(m.group(1)), float(m.group(2))
    if os.path.exists(js):
        d = json.load(open(js))
        out['intr_ep'] = d.get('too_close_ratios', [])
    out['ok'] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='logs')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--n', type=int, default=3500)
    a = ap.parse_args()

    seeds = [read_seed(a.logs, s) for s in a.seeds]
    done = [s for s in seeds if s['ok']]
    print('=' * 78)
    print('  DEPLOYABLE SYSTEM -- predict_method=none -- SELF-INTERPRETING REPORT')
    print('=' * 78)
    for s in seeds:
        if s['ok']:
            print('  seed %d: %d/%d successes = %.2f%%   intrusion %.4f +/- %.4f'
                  % (s['seed'], s['succ'], a.n, 100 * s['succ'] / a.n,
                     s.get('intr_mean', float('nan')), s.get('intr_sd', float('nan'))))
        else:
            print('  seed %d: NOT AVAILABLE (%s)' % (s['seed'], s.get('error', '?')))
    if not done:
        print('\n  No completed seeds. Nothing to analyse yet.')
        return 1
    if len(done) < len(seeds):
        print('\n  WARNING: only %d of %d seeds complete. Results below are PROVISIONAL.'
              % (len(done), len(seeds)))

    # ------------------------------------------------------------ A. success rate
    print('\n' + '-' * 78)
    print('  A. SUCCESS RATE vs HEIGHT (88.03%)')
    print('-' * 78)
    rates = [100 * s['succ'] / a.n for s in done]
    mean = sum(rates) / len(rates)
    sd = math.sqrt(sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)) if len(rates) > 1 else float('nan')
    print('  mean +/- SD over %d seeds : %.2f +/- %.2f %%' % (len(done), mean, sd))
    print('  HEIGHT                    : %.2f %%' % (100 * HEIGHT_SUCC / HEIGHT_N))
    print('  ORACLE (truth futures)    : %.2f +/- %.2f %%' % (ORACLE_SUCC_MEAN, ORACLE_SUCC_SD))
    if len(done) > 1 and sd > 0:
        t = (mean - 100 * HEIGHT_SUCC / HEIGHT_N) / (sd / math.sqrt(len(done)))
        print('  one-sample t vs HEIGHT    : t=%.2f (df=%d)' % (t, len(done) - 1))
    pooled = sum(s['succ'] for s in done)
    sd_d, sd_lo, sd_hi, sd_z, sd_p = two_prop(pooled, a.n * len(done), HEIGHT_SUCC, HEIGHT_N)
    print('  pooled z-test vs HEIGHT   : %+.2f pp  CI[%+.2f, %+.2f]  z=%.2f  p=%.3f'
          % (sd_d, sd_lo, sd_hi, sd_z, sd_p))
    print('    (pooled test treats %d episodes from %d policies as one sample: it is'
          % (a.n * len(done), len(done)))
    print('     PSEUDO-REPLICATED and is shown for reference only; the verdict below uses')
    print('     the per-seed tests, each n=%d vs n=%d.)' % (a.n, HEIGHT_N))
    per_seed_succ = []
    for s in done:
        d1, lo1, hi1, z1, p1 = two_prop(s['succ'], a.n, HEIGHT_SUCC, HEIGHT_N)
        per_seed_succ.append((s['seed'], d1, p1))
        print('  seed %d vs HEIGHT          : %+.2f pp  CI[%+.2f, %+.2f]  z=%.2f  p=%.3f'
              % (s['seed'], d1, lo1, hi1, z1, p1))
    cost = ORACLE_SUCC_MEAN - mean
    print('  privileged info was worth : %+.2f pp  (oracle minus deployable)' % cost)
    # Route on SIGNIFICANCE, not a fixed threshold, and on PER-SEED tests, not the pooled
    # one. "Significantly below" requires a MAJORITY of seeds individually below HEIGHT at
    # p<0.05; one bad seed out of three is reported, not routed on.
    n_below = sum(1 for (_, d1, p1) in per_seed_succ if p1 < 0.05 and d1 < 0)
    succ_sig_worse = n_below * 2 > len(per_seed_succ)
    succ_ok = not succ_sig_worse
    if 0 < n_below and not succ_sig_worse:
        print('  NOTE: %d of %d seeds individually below HEIGHT at p<0.05 -- report per seed.'
              % (n_below, len(per_seed_succ)))
    if succ_ok:
        succ_verdict = ('MATCHES HEIGHT (%d/%d seeds significantly below; pooled %+.2f pp shown '
                        'for reference only)' % (n_below, len(per_seed_succ), sd_d))
    else:
        succ_verdict = ('SIGNIFICANTLY BELOW HEIGHT (%d/%d seeds significantly below; pooled %+.2f pp)'
                        % (n_below, len(per_seed_succ), sd_d))
    print('  >> VERDICT A: %s' % succ_verdict)

    # ------------------------------------------------------------ B. intrusion ratio
    print('\n' + '-' * 78)
    print('  B. INTRUSION RATIO vs HEIGHT (5.6578 +/- 7.0961) -- lower is better')
    print('-' * 78)
    intr_done = [s for s in done if 'intr_ep' in s and s['intr_ep']]
    intr_verdict = 'NOT TESTABLE (no per-episode intrusion data in logs)'
    intr_ok = False
    if intr_done:
        allep = [x for s in intr_done for x in s['intr_ep']]
        m = sum(allep) / len(allep)
        sdv = math.sqrt(sum((x - m) ** 2 for x in allep) / (len(allep) - 1))
        diff, lo, hi, t, pv, dd = welch(m, sdv, len(allep), HEIGHT_INTR_MEAN, HEIGHT_INTR_SD, HEIGHT_N)
        rel = 100 * diff / HEIGHT_INTR_MEAN
        print('  pooled over %d seeds (%d episodes): %.4f +/- %.4f' % (len(intr_done), len(allep), m, sdv))
        print('  HEIGHT                              : %.4f +/- %.4f' % (HEIGHT_INTR_MEAN, HEIGHT_INTR_SD))
        print('  ORACLE (truth futures)              : %.4f  (was -16.1%% vs HEIGHT)' % ORACLE_INTR_MEAN)
        print('  difference                          : %+.4f  (%+.1f%% relative)' % (diff, rel))
        print('  Welch t / p / Cohen d               : t=%.2f  p=%.2e  d=%.3f' % (t, pv, dd))
        print('    (pooled Welch is pseudo-replicated across policies; verdict uses per-seed tests)')
        per_seed = []
        n_sig_better = 0
        for s in intr_done:
            ep = s['intr_ep']; ms = sum(ep) / len(ep)
            ss = math.sqrt(sum((x - ms) ** 2 for x in ep) / (len(ep) - 1))
            d1, lo1, hi1, t1, p1, dd1 = welch(ms, ss, len(ep), HEIGHT_INTR_MEAN, HEIGHT_INTR_SD, HEIGHT_N)
            per_seed.append(ms)
            if p1 < 0.05 and d1 < 0:
                n_sig_better += 1
            print('    seed %d mean %.4f  (%+.1f%% vs HEIGHT)  Welch t=%.2f p=%.1e d=%.3f'
                  % (s['seed'], ms, 100 * (ms - HEIGHT_INTR_MEAN) / HEIGHT_INTR_MEAN, t1, p1, dd1))
        majority = n_sig_better * 2 > len(intr_done)
        if majority and diff < 0:
            if rel <= -12:
                intr_verdict = 'ADVANTAGE SURVIVES (%.0f%% fewer intrusions; %d/%d seeds p<0.05; pooled d=%.2f)' % (-rel, n_sig_better, len(intr_done), dd)
                intr_ok = True
            else:
                intr_verdict = 'ADVANTAGE SHRUNK but significant (%.0f%% fewer; %d/%d seeds p<0.05; d=%.2f)' % (-rel, n_sig_better, len(intr_done), dd)
                intr_ok = True
        elif diff < 0:
            intr_verdict = 'ADVANTAGE GONE (%.0f%% fewer, but only %d/%d seeds significant)' % (-rel, n_sig_better, len(intr_done))
        else:
            intr_verdict = 'REVERSED (%.0f%% MORE intrusions)' % rel
        print('  >> VERDICT B (vs the released checkpoint, confounded by training length): %s'
              % intr_verdict)
    else:
        print('  ' + intr_verdict)

    # ------------------------------------------------------------ B2. compute-matched
    print('\n' + '-' * 78)
    print('  B2. COMPUTE-MATCHED CONTROL: nopredict-350M vs HEIGHT fine-tuned to 350M')
    print('-' * 78)
    ft = read_heightft(a.logs)
    ft_line = ''
    ft_rel = 0.0
    matched_tested = False
    matched_ok = False
    if not ft['ok']:
        print('  %s not found or incomplete -- the released-checkpoint comparison above is'
              % os.path.join(a.logs, HEIGHTFT_LOG))
        print('  then NOT compute-matched (HEIGHT 199.8M vs this arm 350M). State that.')
    else:
        print('  HEIGHT-350M success : %d/%d = %.2f%%' % (ft['succ'], HEIGHT_N, 100 * ft['succ'] / HEIGHT_N))
        for s in done:
            d1, lo1, hi1, z1, p1 = two_prop(s['succ'], a.n, ft['succ'], HEIGHT_N)
            print('    seed %d vs HEIGHT-350M : %+.2f pp  z=%.2f  p=%.3f' % (s['seed'], d1, z1, p1))
        if 'intr_ep' in ft and ft['intr_ep'] and intr_done:
            fe = ft['intr_ep']; fm = sum(fe) / len(fe)
            fs = math.sqrt(sum((x - fm) ** 2 for x in fe) / (len(fe) - 1))
            print('  HEIGHT-350M intrusion: %.4f +/- %.4f  (released HEIGHT was %.4f: %+.1f%% from extra compute alone)'
                  % (fm, fs, HEIGHT_INTR_MEAN, 100 * (fm - HEIGHT_INTR_MEAN) / HEIGHT_INTR_MEAN))
            n_sig = 0
            for s in intr_done:
                ep = s['intr_ep']; ms = sum(ep) / len(ep)
                ss = math.sqrt(sum((x - ms) ** 2 for x in ep) / (len(ep) - 1))
                d1, lo1, hi1, t1, p1, dd1 = welch(ms, ss, len(ep), fm, fs, len(fe))
                if p1 < 0.05 and d1 < 0:
                    n_sig += 1
                print('    seed %d vs HEIGHT-350M : %.4f (%+.1f%%)  Welch t=%.2f p=%.1e d=%.3f'
                      % (s['seed'], ms, 100 * (ms - fm) / fm, t1, p1, dd1))
            print('  >> VERDICT B2: %d/%d seeds intrude significantly less than the COMPUTE-MATCHED HEIGHT.'
                  % (n_sig, len(intr_done)))
            matched_tested = True
            matched_ok = n_sig * 2 > len(intr_done)
            if matched_ok:
                print('     The intrusion advantage is NOT explained by compute. This is the sentence to write.')
            else:
                print('     Against matched compute the advantage does not hold; the released-checkpoint')
                print('     comparison was confounded by training length. Say so; do not claim it.')
            ft_rel = 100 * (sum(allep) / len(allep) - fm) / fm
            ft_line = ('HEIGHT fine-tuned to a matched $350$\\,M steps reaches $%.2f\\%%$ success and an '
                       'intrusion ratio of $%.3f$; %d of %d seeds of the deployable system intrude '
                       'significantly less than this compute-matched control.'
                       % (100 * ft['succ'] / HEIGHT_N, fm, n_sig, len(intr_done)))
        else:
            print('  (no per-episode intrusion JSON for HEIGHT-350M; success comparison only)')

    # The compute-matched control decides the intrusion question whenever it is available:
    # the released checkpoint trained for 199.8M steps, this arm for 350M, so verdict B alone
    # is confounded by training length.
    if matched_tested:
        intr_ok = matched_ok
        released_part = intr_verdict
        intr_verdict = ('%s AT MATCHED COMPUTE (%d/%d seeds intrude significantly less than '
                        'HEIGHT-350M) [against the released checkpoint alone: %s]'
                        % ('ADVANTAGE HOLDS' if matched_ok else 'NO ADVANTAGE', n_sig,
                           len(intr_done), released_part))
        print('\n  >> INTRUSION DECISION (uses B2): %s' % intr_verdict)

    # ------------------------------------------------------------ C. overall
    print('\n' + '=' * 78)
    print('  OVERALL VERDICT AND WHAT TO DO')
    print('=' * 78)
    if succ_ok and intr_ok:
        branch = 1
        print('  BRANCH 1: success not significantly below HEIGHT AND intrusion advantage survives.')
        print('  -> This is the paper\'s HEADLINE competitive claim. Promote to abstract.')
        print('  -> "A deployable system that matches HEIGHT on success and reduces')
        print('      personal-space intrusions by %.0f%% without privileged information."' % (-rel))
    elif succ_ok and not intr_ok:
        branch = 2
        print('  BRANCH 2: success not significantly below HEIGHT but intrusion advantage is gone/n.s.')
        print('  -> The paper stands as a DIAGNOSIS. Remove the "positive result" framing')
        print('     from Sec. XI; keep Table XI as an ORACLE row labelled as such.')
        print('  -> The 16%% was privileged-information-assisted; say so plainly.')
    else:
        branch = 3
        print('  BRANCH 3: success SIGNIFICANTLY below HEIGHT.')
        print('  -> Report honestly. The diagnosis (reciprocity, taxonomy, oracle) is')
        print('     UNAFFECTED -- none of it depends on this run. Drop the competitive claim.')
        print('  -> State: "removing privileged futures costs %.2f pp of success".' % cost)

    # ------------------------------------------------------------ D. LaTeX to paste
    print('\n' + '=' * 78)
    print('  LaTeX FOR THE \\pending BLOCKS -- paste into paper3.tex')
    print('=' * 78)
    seedcells = ' & '.join('%.2f\\%%' % r for r in rates) + (' & --- ' * (3 - len(rates)))
    intrcells = ' & '.join('%.3f' % (sum(s['intr_ep']) / len(s['intr_ep'])) for s in intr_done) if intr_done else '--- & --- & ---'
    print(r'''
%% --- Table: deployable system (replaces the \pending table in Sec. XII) ---
\begin{table}[t]
\caption{Deployable system, \texttt{predict\_method\,=\,'none'}, $n=3{,}500$ per
seed. No privileged information. HEIGHT: $88.03\%%$ success, intrusion ratio $5.66$.}
\label{tab:deployable}
\centering
\begin{tabular}{lcccc}
\toprule
 & Seed 1 & Seed 2 & Seed 3 & Mean $\pm$ SD \\
\midrule
Success & %s & $%.2f \pm %.2f\%%$ \\
Intrusion ratio & %s & %s \\
\bottomrule
\end{tabular}
\end{table}''' % (seedcells, mean, sd, intrcells,
                 ('$%.3f$' % (sum(allep) / len(allep))) if intr_done else '---'))

    print(r'''
%% --- Abstract sentence (replaces the \pending in the abstract) ---''')
    if branch == 1:
        print(r'''Retrained without privileged futures, the system matches HEIGHT on success
($%.2f \pm %.2f\%%$) and reduces personal-space intrusions by $%.0f\%%$ against HEIGHT's
released checkpoint ($p<10^{-%d}$)%s.'''
              % (mean, sd, -rel, max(1, int(-math.log10(pv))),
                 ('' if not ft_line else
                  r', and by %.0f\%% against a compute-matched baseline' % (-ft_rel))))
    elif branch == 2:
        print(r'''Retrained without privileged futures, the system matches HEIGHT on success
($%.2f \pm %.2f\%%$); the intrusion advantage does not survive, and we report it as an
oracle-assisted effect.''' % (mean, sd))
    else:
        print(r'''Retrained without privileged futures, success falls to $%.2f \pm %.2f\%%$;
the diagnostic findings are unaffected.''' % (mean, sd))

    print(r'''
%% --- Sec. XII body paragraph ---
The same architecture retrained with \texttt{predict\_method\,=\,'none'}---no
trajectory prediction of any kind, occlusion inference retained---for $350$\,M steps
across three seeds reaches $%.2f \pm %.2f\%%$ success (Table~\ref{tab:deployable}),
against HEIGHT's $88.03\%%$ (per-seed $p=%s$). The oracle-assisted arm, with privileged
ground-truth futures, differs from this arm by $%+.2f$\,pp of success, a contrast that also
mixes in its longer training (${\sim}426$\,M steps against this arm's $350$\,M).''' % (mean, sd,
          ', '.join('%.2f' % two_prop(s['succ'], a.n, HEIGHT_SUCC, HEIGHT_N)[4] for s in done), cost))
    if intr_done:
        print(r'''The per-seed intrusion ratios are $%s$ against HEIGHT's $5.658$ (per-seed Welch
$p=%s$).''' % (', '.join('%.3f' % (sum(s['intr_ep']) / len(s['intr_ep'])) for s in intr_done),
               ', '.join(ltx_p(welch(sum(s['intr_ep']) / len(s['intr_ep']),
                                     math.sqrt(sum((x - sum(s['intr_ep']) / len(s['intr_ep'])) ** 2 for x in s['intr_ep']) / (len(s['intr_ep']) - 1)),
                                     len(s['intr_ep']), HEIGHT_INTR_MEAN, HEIGHT_INTR_SD, HEIGHT_N)[4]) for s in intr_done)))
    if ft_line:
        print(ft_line)

    # ------------------------------------------------------------ write verdict file
    outp = os.path.join(a.logs, 'NOPREDICT_VERDICT.txt')
    with open(outp, 'w') as f:
        f.write('BRANCH %d\nA: %s\nB: %s\nsuccess mean %.2f sd %.2f\n' % (branch, succ_verdict, intr_verdict, mean, sd))
    print('\n  verdict written to %s' % outp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
