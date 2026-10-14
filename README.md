# EEG × Deep Learning research assignment

Task 1 (motor imagery, PhysioNet EEGBCI subjects 70–109) is complete in this repository. Task 2 (sleep staging) has the original `task2/sleep_pipeline.py` committed unmodified; its analysis has not been started.

## Run

```bash
pip install -r requirements.txt          # Python 3.13; exact versions used for every output
python main.py                           # everything, in order; resumes finished training jobs
python main.py --stages a1,a2,a3         # a subset: a1 a2 a3 b2train b23 b24 b25 b26
```

EDF files are read from `../files` (the PhysioNet folder next to this repository) or from `$EEGBCI_DIR`; any file not found locally is downloaded through `mne.datasets.eegbci`. Trial data and trained weights are cached in `cache/` (not in git).

**Run time on a 2-core CPU:** Part A ≈ 20 min; Part B training ≈ 7 h (≈ 240 training runs of 200–500 steps, ~1–2 min each). Training jobs are written to their CSV as they finish, so an interrupted run continues where it stopped.

## Where things are

| | |
|---|---|
| `task1/audit.py` | A1 data audit → `outputs/task1/audit.csv` |
| `task1/data.py` | shared trial loader (filtering, epoching, rejection) |
| `task1/a2_erd.py` | A2 ERD, lateralisation index, permutation test, figures |
| `task1/a3_confounds.py` | A3 route decoders, chance thresholds, per-subject table |
| `task1/models.py` | EEGNet (80 Hz) and the transformer with three tokenisers |
| `task1/train.py` | folds, preprocessing, training, resumable runner |
| `task1/b2_*.py` | B2 experiments 1–6 |
| `outputs/task1/` | every CSV and figure quoted in the write-up |
| `outputs/task1/WRITEUP.md` | Task 1 write-up |
| `PREDICTIONS.md` | each prediction, committed before its result, with the outcome added afterwards |
| `DECISIONS.md` | every non-trivial choice: choice / alternative / why / what would change |

## Seeds
Project seed 42 (`main.py`). Per-run seeds are derived from it (DECISIONS D0, D25). Every seed's result is in the run CSVs (`b2_runs.csv`, `b2_5_runs.csv`, `b2_6_runs.csv`), one row per test subject.
