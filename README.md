# Learned Acoustic Signalling for Occlusion-Aware Robot Navigation in Dense Crowds

Supporting material for **Paper 1: the five-page reinforcement-learning lab report,
version 2, 22 October 2026**, by Ashwinder Pal Singh. Read
[the exact report PDF](report/paper1_lab_report.pdf).

The robot learns whether to beep while navigating. Beep-only and beep-plus-listening
policies are compared with the same platform without acoustics. All combined policies
use privileged simulator forecasts; these experiments do not demonstrate a deployable
predictor or an overall architectural advantage over HEIGHT.

| Setting | HEIGHT | No acoustics | Beep only | Beep + listening | Without confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nominal | 88.03% | 88.54% | 89.49% | 89.60% | 88.57% |
| 10–14 pedestrians | 77.94% | 78.97% | 81.51% | 80.14% | Not evaluated |
| 13–17 obstacles | 84.89% | 84.83% | 86.46% | Not evaluated | Not evaluated |

Each entry is one checkpoint on 3,500 episodes. Beep-only improves on the matched
no-acoustics arm by 2.54 points in the denser crowd (unadjusted p=0.00755). Listening
does not add a detected success gain. The report uses a conservative Bonferroni bar
of p<0.004; the denser-crowd beep-only comparison with released HEIGHT passes it,
whereas its comparison with no acoustics does not. The three-scene-set comparison
reuses the same trained checkpoints and does not measure training-seed variability.

## Contents

| Path | Purpose |
| --- | --- |
| `report/` | Unchanged specified PDF, LaTeX source and its two PDF figures; the acoustic diagram is drawn in LaTeX |
| `analysis/` | Recompute report statistics and the results figure; optional CPU checkpoint inspection/evaluation |
| `results/logs/` | The 16 primary evaluation logs and four acoustic/base fine-tuning logs |
| `results/development/` | Three n=500 development logs cited in the development history; not final evaluation evidence |
| `results/training/` | Shared Stage 2 training and teacher/data-collection provenance |
| `results/runtime/` | Historical DGX package listing, for provenance rather than portable installation |
| `configs/runs/` | Archived configs for the five evaluated arms and shared Stage 2 checkpoint |
| `checkpoints/` | Those five policy checkpoints, shared Stage 2 weights and frozen PaS teacher |
| `source/` | Simulator, acoustic mechanism, policy, PPO, occlusion training and their imported upstream dependencies |
| `PROVENANCE.md` | Map report results to evidence and explain archived-config limitations |
| `CHECKSUMS.sha256` | SHA-256 manifest of the tracked artifact files, excluding this manifest itself |

The current `main` tree is scoped to this lab report. Git history retains the previous
reviewer snapshot. Journal-only diagnostic datasets, analyses, manuscripts and figures
are excluded from the current tree.

## Recompute the reported results

The numerical analysis needs only Python's standard library:

```bash
python3 -I -B analysis/compute_stats.py
```

It tallies all 3,500 terminal outcomes in each primary log, checks them against the
logged success count, records input hashes in `analysis/stats.json`, and computes
the reported success contrasts, confidence intervals and scene-set replication.
Navigation-time t statistics use the rounded logged summaries; the script labels its
normal-approximation p values explicitly. The report quotes the t statistics.

To regenerate the results figure with Matplotlib:

```bash
mkdir -p work
python3 -I -B analysis/make_figure.py work/fig_results.pdf
```

The original figure stays in `report/`.

Verify the original files before recomputing derived outputs:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

## Optional simulator setup and checkpoint use

The historical DGX used Python 3.10.12 in NVIDIA NGC 24.06, NVIDIA PyTorch
`2.4.0a0+f70bd71a48.nv24.6`, Gym 0.22.0, NumPy 1.24.4, PyBullet 3.2.6,
pyglet 1.5.27 and a 40 GB MIG allocation on an H100. The package listing is
`results/runtime/pip_freeze_dgx_20261012.txt`; its local wheel paths are not an
installable requirements file. A portable CPU installation is:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r source/requirements.txt
.venv/bin/python -m pip install 'git+https://github.com/sybrenstuvel/Python-RVO2.git@c2c46ba8d59556aa10faf03479293236efea154d'
```

RVO2 needs a C++ compiler and CMake. `baselines/`, `pybullet_envs/` and
`roboschool.py` under `source/` provide the small compatibility modules used by
this checkout. The unrelated TensorFlow dependency from the old requirements
has been removed. The CPU recipe is not a claim of identical native-library
builds or exact historical trajectory reproduction.

Load a checkpoint strictly and inspect its parameter count without evaluating:

```bash
.venv/bin/python -B analysis/evaluate_arm.py --arm nospeak --inspect --out work/inspect_nospeak
```

Run a short CPU smoke evaluation, or explicitly request the full cell:

```bash
.venv/bin/python -B analysis/evaluate_arm.py --arm full --episodes 2 --out work/full_smoke
.venv/bin/python -B analysis/evaluate_arm.py --arm noreturn --setting humans12 --episodes 3500 --out work/beep_dense
```

Arms are `height`, `nospeak`, `noreturn` (beep only), `full` and `nocal`.
Settings are `nominal`, `humans12` and `obstacles15`. `--scene-seed 2` or `3`
selects the additional nominal scene sets for `full` and `nospeak` only. The
helper rejects combinations absent from the report and existing output folders.
It uses one CPU worker, strictly loads every policy tensor, reads archived configs
and weights, and records explicit effective settings under the new output directory.
New smoke outputs do not replace the manuscript's historical evidence.

Training source is retained in `source/train.py`; the exact archived configs and
shared Stage 2 weights document the 50 M-step fine-tuning setup. The migration
utility is `source/migrate_weights_stage3.py`. No training is launched by the
analysis or evaluation commands above. Long training is not needed to recompute
the paper's tables from its archived observations.

## Limitations and attribution

Every acoustic arm uses privileged ORCA futures. Pedestrian beep compliance is
a chosen simulator rule, not measured human behavior. Only one training run per
arm was studied; no real-robot validation, inference latency or power measurement
is claimed. The archived evaluator retains the historical episode-reset and
terminal-step metric behavior; see `PROVENANCE.md` before interpreting fresh runs.

The navigation platform derives from [HEIGHT](https://github.com/Shuijing725/CrowdNav_HEIGHT),
with intention and PaS occlusion components described and cited in the report.
The upstream MIT notice is preserved in `LICENSE`.
