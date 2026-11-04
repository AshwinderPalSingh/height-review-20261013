# Evidence for the specified Paper 1 lab report

Scope: `paper1_lab_report.pdf`, version 2 of 22 October 2026.
SHA-256: `70b6cc19403539aec9afd8e49f3aab81292f4f458253ac1fb38711e6602bdd9c`.
The PDF, LaTeX and original figures are unchanged copies. Its existing bibliography
points to the earlier public source repository; this repository now holds a scoped
supporting artifact, without rewriting that fixed report.

## Primary evaluation inputs

All names in this table are under `results/logs/`. `analysis/compute_stats.py`
reads exactly these 16 evaluation logs. All contain 3,500 terminal outcomes.

| Report row or comparison | Log |
| --- | --- |
| Table I, released HEIGHT | `eval_heightpub.log` |
| Table I, no acoustics | `eval_exp_nospeak_s1.log` |
| Table I, beep only | `h1_exp_noreturn_s1_n3500.log` |
| Table I, beep + listening | `h1_exp_full_s1_n3500.log` |
| Table I, without confidence | `eval_nocal_s1_n3500.log` |
| Full, scene set 2 | `evalseed2_exp_full_s1.log` |
| Full, scene set 3 | `evalseed3_exp_full_s1.log` |
| No acoustics, scene set 2 | `evalseed2_exp_nospeak_s1.log` |
| No acoustics, scene set 3 | `evalseed3_exp_nospeak_s1.log` |
| Table II, denser HEIGHT | `gen_more_human_heightpub.log` |
| Table II, denser no acoustics | `gen_more_human_nospeak.log` |
| Table II, denser beep only | `gen_more_human_noreturn.log` |
| Table II, denser beep + listening | `gen_more_human_full.log` |
| Table II, more obstacles, HEIGHT | `gen_more_obs_heightpub.log` |
| Table II, more obstacles, no acoustics | `gen_more_obs_nospeak.log` |
| Table II, more obstacles, beep only | `gen_more_obs_noreturn.log` |

Figure 3 uses the same counts, via `analysis/stats.json` and `make_figure.py`.
Intrusion, acoustic counts and successful-episode navigation summaries come from
these logs. No journal-only trajectory dump is needed for these report values.

## Checkpoints and configs

| Report role | Config under `configs/runs/` | Weight under `checkpoints/` |
| --- | --- | --- |
| HEIGHT reference | `exp_heightpub/config.py` | `exp_heightpub/237800.pt` |
| No acoustics | `exp_nospeak_s1/config.py` | `exp_nospeak_s1/59522.pt` |
| Beep only | `exp_noreturn_s1/config.py` | `exp_noreturn_s1/59522.pt` |
| Beep + listening | `exp_full_s1/config.py` | `exp_full_s1/59522.pt` |
| Without confidence | `exp_nocal_s1/config.py` | `exp_nocal_s1/59522.pt` |
| Shared fine-tuning parent | `stage2_HEIGHT_intention_occlusion/config.py` | `stage2_HEIGHT_intention_occlusion/357141.pt` |

`occlusion_vae/label_vae_weight_300.pth` is the frozen PaS teacher; it is excluded
from the report's trainable actor–critic counts. Table III counts and checkpoint
sizes can be checked by `analysis/evaluate_arm.py --inspect` for `height` and
`nospeak`; the no-acoustics training log also prints 1,236,058 trainable parameters.
Exact weight hashes are in `CHECKSUMS.sha256`.

The three acoustic checkpoints and configs were imported from the project's Drive
backup `HEIGHT_Backup/checkpoints/<run>/` and `HEIGHT_Backup/data_meta/<run>/configs/`.
Their recorded config bytes are preserved. Saved configs were edited during past
testing: the full arm's snapshot now has `env.seed=3`, and reference/shared-parent
snapshots retain other historical scene seeds. They are not immutable original
training-config captures. The new evaluation helper records and explicitly sets
scene seed 1 for the nominal reported cell, or 2/3 for the additional test sets.
It also replaces machine-relative media/teacher/output paths in memory, selects
CPU and disables rendering, without editing the saved snapshots.

The full and no-acoustics checkpoints are reused across scene sets and settings.
These are repeated evaluations of fixed policies, not independent training runs.
Full with more obstacles and the confidence ablation in either harder setting
were not evaluated in the report; no corresponding result is supplied.

## Training and development history

Four fine-tuning logs are in `results/logs/exp_{nospeak,full,noreturn,nocal}_s1_train.log`.
`results/training/` contains shared Stage 2 training, collection and teacher-training
logs. These support the shared-parent and training-budget account; the repository
does not contain every intermediate optimizer checkpoint or teacher training grid.
The three `results/development/eval_pathb_{fixed,v2,v3}.log` files support the
reported rounded n=500 development values (90%, 77%, 90%). They are historical
development evidence and must not be pooled into the primary evaluation.

## Source and execution limits

The source derives from the earlier reviewer snapshot. The optional shield and
offline trajectory-recording branches were removed from the evaluation harness
because they do not support this report. They were disabled in the report's runs;
policy, simulator, PPO and acoustic/metric calculations are retained. Upstream
model/environment modules imported by the simulator are supporting dependencies,
not additional experimental claims. No journal-only analysis scripts or datasets
are included in the current tree.

The historical evaluator can carry robot command state between episodes and its
path-length summary includes an auto-reset jump. These behaviors are preserved
rather than silently changing the contract behind archived results. Path length
is not a reported lab-report endpoint. Fresh simulator execution on different
native builds is not guaranteed to reproduce every archived action or trajectory.
Recomputing the reported statistics from the supplied logs does not depend on
the historical simulator runtime.
