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

### D7. Window −2 … +3.975 s around the cue
- **Choice:** trials from 2 s before to 3.975 s after the cue (957 samples).
- **Why:** the 4.1 s rest that precedes every cue holds the baseline; the 4.1 s cue period holds the imagery. **Own inspection:** in many runs the last cue starts at 119.0 s and the file ends at 122.994 s, so a +4.0 s window silently dropped that last trial in 6 subjects (MNE drop log "TOO_SHORT"); +3.975 s keeps them. Only S104's genuinely truncated trial (D4) is lost.
- **With the alternative:** a shorter pre-cue window (e.g. −1 s) would leave no room for a baseline that excludes cue-anticipation effects.

### D8. Trial rejection relative to each subject (3 × median p-p, ocular channels excluded)
- **Choice:** reject a trial if its largest non-Fp/AF peak-to-peak exceeds 3× that subject's median.
- **Alternative:** a fixed threshold such as 100–500 µV.
- **Why:** **own inspection:** amplitude differs 5–8× between subjects on every channel (median C3 p-p ≈ 60 µV in S072 vs ≈ 530 µV in S075; same at 1–40 Hz). A fixed 500 µV threshold would reject 60–100% of trials in 19 subjects. About 25–60% of each channel's variance is common to all 64 channels (see D9), which points to the undocumented reference. Result: **0 of 1,662 trials rejected**; within a subject no trial is an outlier by this rule.
- **With the alternative:** a fixed threshold would remove almost all trials of the high-amplitude subjects and keep everyone else's, which turns amplitude scale into a subject-selection rule.

---

## A2 — Physiological ground truth (`task1/a2_erd.py`)

### D9. Spatial reference: surface Laplacian (CSD), with CAR and raw as sensitivity checks
**What the missing reference means for C3 vs C4.** Each recorded channel is V(electrode) − V(ref). Power at "C3" is the power of C3 − ref, which contains the reference's own activity and a cross-term between the two. Three consequences:
1. If the reference carries mu/beta (e.g. a vertex or ear site picking up sensorimotor rhythm), its ERD appears at both C3 and C4 and shrinks their difference.
2. If the reference is lateral (one ear or mastoid), it adds more to one hemisphere's channels than the other, so a C3−C4 power difference can appear or flip for reasons unrelated to the brain.
3. Amplitude scale between subjects is set partly by the reference, so absolute power cannot be compared across subjects.

**Own inspection supports this:** 25–60% of each channel's variance is common to all 64 channels (S070 0.60, S072 0.60, S074 0.61, S079 0.25, S085 0.41); C3's SD falls from 48 → 18 µV (S070) and 15 → 6 µV (S072) after removing the common average.

- **Choice:** spherical-spline surface Laplacian (current source density; MNE defaults, spline order 4, λ = 1e-5) on all 64 channels, for every A2 number.
- **Alternatives:** common average reference (CAR); keep the raw (unknown) reference.
- **Why:** CSD is reference-free: it estimates the local radial current under each electrode from the second spatial derivative, so any signal shared by all channels (including the reference) cancels exactly. It also sharpens the spatial focus to roughly the electrode's own neighbourhood, which is what a C3-vs-C4 comparison needs. CAR also removes the shared signal, but it subtracts the mean of 64 channels that include the frontal ocular channels, so a large blink leaks into every channel with the opposite sign.
- **What changes with the alternatives (measured):** lateralised ERD present in **18/37 (CSD), 15/37 (CAR), 12/37 (raw)**; labels agree with CSD for 81% (CAR) and 78% (raw) of subjects. Median contralateral mu ERD shrinks from −1.35 dB (CSD) to −0.80 dB (raw). The raw reference hides a third of the lateralised subjects. Figure: `figures/a2_reference_effect.png`.

### D10. Baseline −1.5 to −0.5 s, pooled over both classes
- **Choice:** mean power from −1.5 to −0.5 s before the cue, averaged over all trials of the subject (both classes), per channel and frequency.
- **Alternatives:** −2 to 0 s; a per-trial baseline; a per-class baseline.
- **Why:** the 4.1 s rest that precedes each cue starts at −4.1 s. −1.5 s is 2.6 s after the previous imagery ended, past the post-imagery beta rebound (≈1–2 s); −0.5 s keeps the 0.5 s Morlet wavelets from smearing the cue response into the baseline, and the cue rhythm is fixed (every 4.1 s), so anticipatory ERD could start just before 0. Pooling over trials avoids the noise of single-trial baselines; pooling over classes guarantees the baseline cannot itself create a left/right difference.
- **With the alternatives:** ending the baseline at 0 s would let the post-cue drop leak into it and shrink the ERD; a per-class baseline could manufacture lateralisation from pre-cue differences.

### D11. Time–frequency: Morlet wavelets, 4–40 Hz, n_cycles = f/2, ERD window 0.5–3.975 s
- **Choice:** Morlet power on 11 channels (FC3, C5, C3, C1, CP3, Cz, FC4, C6, C4, C2, CP4); 0.5 s wavelets at every frequency; band power averaged over the ERD window starting 0.5 s after the cue.
- **Why:** the 0.5 s wavelet gives ~2 Hz resolution in mu while keeping onset smearing short. Starting at 0.5 s skips the visual evoked response to the target appearing (itself an A3 confound route).
- **Bias found and fixed:** the first version averaged single-trial dB values. Since E[log P] < log E[P] for skewed power, that created a fake −2.5 dB "ERD" over the whole map, including the baseline. Band power is now averaged over frequency and time **before** taking dB, and the figure maps average power over trials before taking dB.

### D12. Lateralisation index, test and label
- **Index:** for each trial, e_ch = 10·log10(band power in ERD window / subject baseline band power); h = e_C4 − e_C3; **LI = mean(h | right) − mean(h | left)** in dB. Contralateral ERD predicts LI > 0 (right hand → more ERD at C3 → h larger).
- **Test:** one-sided label-permutation test within the subject, 10,000 permutations, seed = project seed + subject number; p = (1 + #null ≥ observed)/(10,001).
- **Label "present":** for at least one band, p < 0.025 (two bands, Bonferroni, family-wise α = 0.05) **and** the contralateral band power falls below baseline (mean of C4 on left trials and C3 on right trials < 0 dB). The second condition excludes a lateralisation made only of ipsilateral power increases.
- **Alternatives:** a normalised LI ((C3−C4)/(C3+C4) in linear power), a t-test, a single band, or no ERD-sign condition.
- **Why:** a difference of dB values is already normalised for each subject's power scale and its baseline, the permutation test makes no distributional assumption with ~22 trials per class, and the label needs to say "lateralised **ERD**", not "lateralised power".
- **With the alternatives:** without the Bonferroni split, one more subject would count as present (S076, β p = 0.031). The ERD-sign condition changed no label in this data: every subject with p < 0.025 also had contralateral power below baseline. No correction across the 37 subjects is applied, because each label is a statement about one subject; at α = 0.05 about 1–2 false "present" labels are expected among truly absent subjects.

### D13. Examples shown
- 3 "present" subjects with the smallest p (strongest evidence) and 3 "absent" subjects with the largest p (weakest), picked by that rule in code, not by eye.
