# Lab-report artifact checks, 4 November 2026

- The report PDF is byte-identical to the specified 22 October v2 PDF;
  its source and two figures are also unchanged.
- All 16 primary logs were parsed, totaling 56,000 historical episode records.
  Recomputed `analysis/stats.json` is exactly equal to the original lab-report
  analysis, including input hashes, counts, contrasts and scene-set replication.
- The results figure script ran successfully from the recomputed statistics.
- All Python source and analysis files passed syntax parsing.
- All five evaluated policy checkpoints loaded with `strict=True` in the supplied
  source and each completed two local CPU episodes. HEIGHT has 478,618 trainable
  actor–critic parameters and no-acoustics has 1,236,058, matching Table III.
- On the same CPU runtime, two full-arm episodes produced identical outcome/step
  records, metric summaries and per-episode intrusion arrays with the original
  evaluator and the evaluator with optional shield/trajectory capture removed.
- Every shipped file is listed in `CHECKSUMS.sha256`, except the manifest itself.

Runtime scope: an existing macOS ARM Python 3.10.18 / PyTorch 2.4.0 CPU environment.
This checks loading, execution and the scoped harness edit; it is not a fresh
installation test, a complete 3,500-episode rerun, or exact historical DGX
trajectory reproduction. The manuscript results remain the archived log analysis.
No optimizer updates or GPU training were performed.
