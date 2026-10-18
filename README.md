# EEG × deep learning: motor imagery on PhysioNet EEGBCI (subjects 70–109)

What is in the signal for imagined left vs right fist, and what do EEGNet and a small
transformer learn from it? Short answer: the cue makes subjects **look at the target**, and that
eye movement decodes left/right far better than sensorimotor ERD does; the networks use it.
See `REPORT.pdf` (3 pages, including the 1-page write-up).

## Run

```bash
pip install -r requirements.txt
python main.py                       # everything, in order, then REPORT.pdf
python main.py --stages a2,a3        # a subset: a1 a2 a3 b1 b2train b2report b23 report
```

- EDF files are read from `../files` (the PhysioNet folder next to this repository) or from
  `$EEGBCI_DIR`; anything missing is downloaded with `mne.datasets.eegbci`.
- Epochs, features, trained weights and attributions are cached in `cache/` (not in git); delete
  it to recompute everything from the EDFs.
- Training is resumable (per-epoch checkpoints). `EEG_TIME_BUDGET=<seconds>` makes the
  `b2train` stage stop cleanly and continue on the next call; without it, it runs to the end.
- Seed: 42 everywhere (`main.py`, `task1/train.py`); every run is in `outputs/task1/b2_runs.csv`.
- Run time on a 4-core CPU: Part A ≈ 5 min, Part B training ≈ 40 min, B2.3 ≈ 3 min.

## Reproducibility check

With `cache/` deleted, stages a1, a2, a3, b1 were re-run from the EDF files and reproduced every
committed CSV and figure byte for byte. Two models (`eegnet_n5`, `tf-time_n5`) were retrained from
scratch, one of them deliberately interrupted and resumed, and gave bit-identical weights and
results; the integrated-gradient maps recomputed bit-identically; b2report, b23 and the report
then reproduced all remaining outputs. (The other nine models were not retrained in that check.)

## Where things are

| file | what |
|---|---|
| `task1/data_audit.py` | A1 audit → `audit.csv`, `audit_runs.csv`; refuses unexplained anomalies |
| `task1/data.py` | shared loader: montage, per-run bad-channel interpolation, filtering, epochs |
| `task1/a2_erd.py` | A2 CSD time–frequency ERD, mu lateralisation index, permutation test + FDR |
| `task1/a3_confounds.py` | A3 eye-movement / visual / motor decoders, binomial chance thresholds |
| `task1/models.py` | EEGNet (160 Hz) and the transformer with three tokenisers |
| `task1/train.py` | subject split, train-only normalisation, resumable training |
| `task1/b2_experiments.py` | B1 parameter counts, B2.1 scaling, B2.2 tokenisation |
| `task1/b2_reliance.py` | B2.3 Integrated Gradients, deletion tests, HEOG regression, correlations |
| `task1/report.py` | builds `REPORT.pdf` from the result files |
| `outputs/task1/` | every CSV and figure; `per_subject_table.csv` = mu LI, p, label, A3 accuracies + thresholds |
| `PREDICTIONS.md` | each prediction, committed before its result; outcomes appended afterwards |
| `DECISIONS.md` | every non-trivial choice: choice / alternative / why / what would change |

Analyses added after seeing a result are marked **post-hoc** in the code, CSVs, DECISIONS.md and
the report.

Commits before `31929d4` belong to an earlier version of the assignment and are kept for history.
