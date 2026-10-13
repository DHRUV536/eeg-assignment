# DECISIONS

Every non-trivial choice, in the format: **choice / alternative / why / what would change with the alternative.**
"Verified" says where a dataset fact was checked.

---

## Project-wide

### D0. Seeding and determinism
- **Choice:** one project seed (`SEED = 42` in `main.py`). Every stochastic step derives its seed from it plus a fixed offset per experiment (listed in each module). Python, NumPy and torch are all seeded; `torch.use_deterministic_algorithms(True, warn_only=True)`; training runs on CPU with a fixed thread count.
- **Alternative:** seed only torch, or report a single seed.
- **Why:** the brief asks for all seeds fixed and every run reported. Some CPU kernels are still not bit-exact across machines and thread counts, so results are reported as mean ± spread over seeds, not as bit-exact numbers.
- **With the alternative:** single-seed numbers in this data regime (≈45 trials/subject, ±several % seed-to-seed for the networks) would be hard to tell apart from noise.

---

## A1 — Data audit

### D1. Runs 4, 8, 12 (imagined left vs right fist)
- **Choice:** runs 4, 8 and 12 only.
- **Alternative:** add runs 3, 7, 11 (executed left/right fist; same cue, 2× data), or the hands/feet runs.
- **Why:** the brief asks for imagery only. **Verified:** (1) PhysioNet eegmmidb 1.0.0 page, Methods: runs 1–2 are baselines; "the remaining 12 runs consisted of three repetitions of four two-minute task conditions", in the order execute L/R fist, imagine L/R fist, execute fists/feet, imagine fists/feet, so imagine L/R = runs 4, 8, 12; (2) MNE `mne.datasets.eegbci.load_data` docstring run table: "4, 8, 12  Motor imagery: left vs right hand"; (3) own inspection: runs 4/8/12 of all 40 subjects contain only T0/T1/T2 events with 15 cues per run (`audit_runs.csv`).
- **With the alternative:** execution runs add real movement (EMG, proprioceptive feedback, stronger ERD), so a model trained on them would learn signals that do not exist in imagery; accuracy would rise and would no longer answer the question.

### D2. Event-code mapping depends on the run
- **Choice:** in runs 4/8/12, T1 = imagined left fist (label 0), T2 = imagined right fist (label 1), T0 = rest (not a class).
- **Alternative:** a global T1/T2 mapping.
- **Why:** **Verified** on the PhysioNet page: "T1 corresponds to the onset of … left fist … in unilateral fist runs and to both fists in bilateral hand/foot runs"; T2 likewise right fist / both feet. The same code means different things in different run types, so the mapping is only valid because only unilateral runs are loaded.
- **With the alternative:** mixing in runs 6/10/14 under a global map would label "both fists" as left and "both feet" as right.

### D3. Subjects 88, 92, 100 are dropped (37 subjects kept)
- **Choice:** drop all three.
- **Alternative:** resample 128 → 160 Hz and keep (40 subjects).
- **Why:** these are the only 128 Hz subjects, and **own inspection** of their annotations shows a different trial structure, not just a different rate:

  | | rest before cue | imagery cue | cues/run |
  |---|---|---|---|
  | typical (e.g. S070) | 4.1 s | 4.1 s | 15 |
  | S088, S092 | 1.375 s | 5.125 s | 19 |
  | S100 | 5.125 s | 5.125 s | 12 |

  5.125 = 4.1 × 160/128, so the event timing for these files was produced differently from everyone else's. Resampling changes the sample rate but not the trial timing. In S088/S092 there is only 1.4 s of rest before each cue, so the A2 baseline window (−1.5 to −0.5 s) would fall inside the previous trial.
- **With the alternative:** 3 more subjects (~170 trials), but their trials would mix imagery from the previous cue into the baseline, and the networks would see a different cue length for 8% of subjects.

### D4. S104 run 8: truncated recording, subject kept
- **Choice:** keep; the last trial, cut off at 2.975 s, is lost at epoching because the 4 s window runs past the end of the file.
- **Alternative:** drop the run or the subject.
- **Why:** the run is 106 s instead of 123 s; the other 12 trials are normal. Losing one trial costs less than losing the subject. The loss is visible in the per-subject table (`n_events` vs `n_epoched`).
- **With the alternative:** 1 fewer subject, no change in data quality for the rest.

### D5. High-amplitude frontal channels are flagged, not interpolated
- **Choice:** the audit flags channels whose whole-run peak-to-peak exceeds 1 mV (after a 1 Hz high-pass). These are almost always Fp/AF channels (eye movements and blinks). They are kept as recorded; no channel is interpolated.
- **Alternative:** mark them bad and interpolate (common default).
- **Why:** interpolating Fp/AF from their neighbours would partially erase the ocular signal, and the ocular route is one of the confounds A3 has to measure. Gross artifacts are handled per trial instead (D8).
- **With the alternative:** A3's eye-movement decoder would be weakened by our own preprocessing, and we could wrongly conclude there is no ocular confound.

### D5b. Audit thresholds
- Run duration tolerance 5 s, cue-duration tolerance 0.1 s, flat < 0.5 µV SD, high-amplitude > 1 mV p-p, imbalance |L−R|/(L+R) > 0.2, minimum 15 trials/class. All are set in `task1/audit.py` with the reason next to each. None of them changed which subjects were kept: the only drops come from D3.

---

## Shared preprocessing (`task1/data.py`)

### D6. Notch at 60 Hz, 0.5 Hz high-pass, no low-pass at load time
- **Choice:** per-run 60 Hz notch and 0.5 Hz high-pass; analyses band-limit themselves later.
- **Alternative:** 1–40 Hz band-pass at load time.
- **Why:** **Verified on the data:** the S070 run-4 PSD has a 60 Hz peak 61× the 45–55 Hz median (US mains). The high-pass removes drift but keeps the < 4 Hz ocular signal; leaving content up to 80 Hz keeps the EMG route (> 30 Hz) testable in A3.
- **With the alternative:** a 1–40 Hz load-time filter would make the EMG route untestable and partly hide the ocular route.

### D7. Window −2 … +4 s around the cue
- **Choice:** 6 s trials starting 2 s before the cue.
- **Why:** the 4.1 s rest that precedes every cue holds the baseline; the 4.1 s cue period holds the imagery. 4.0 s rather than 4.1 s keeps one sample of margin.
- **With the alternative:** a shorter pre-cue window (e.g. −1 s) would leave no room for a baseline that excludes cue-anticipation effects.

### D8. Trial rejection relative to each subject (3 × median p-p, ocular channels excluded)
- **Choice:** reject a trial if its largest non-Fp/AF peak-to-peak exceeds 3× that subject's median.
- **Alternative:** a fixed threshold such as 100–500 µV.
- **Why:** **own inspection:** amplitude differs 5–8× between subjects on every channel (median C3 p-p ≈ 60 µV in S072 vs ≈ 530 µV in S075; same at 1–40 Hz). A fixed 500 µV threshold would reject 60–100% of trials in 19 subjects. About 25–60% of each channel's variance is common to all 64 channels (see D9), which points to the undocumented reference. Result: **0 of 1,649 trials rejected**; within a subject no trial is an outlier by this rule.
- **With the alternative:** a fixed threshold would remove almost all trials of the high-amplitude subjects and keep everyone else's, which turns amplitude scale into a subject-selection rule.
