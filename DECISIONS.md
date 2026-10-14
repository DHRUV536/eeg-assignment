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

---

## A3 — Confound audit (`task1/a3_confounds.py`)

### D14. One classifier and one CV scheme for every route
- **Choice:** standardise → shrinkage LDA (Ledoit–Wolf), within-subject stratified 5-fold CV repeated 10× (seed = project seed + subject); accuracy = mean over repeats.
- **Alternatives:** a different classifier per route (e.g. CSP+LDA for power, xDAWN for evoked), or a non-linear classifier.
- **Why:** routes must be comparable with each other, so the only thing that changes between decoders is which signal they see. Shrinkage LDA is stable with 45 trials and 16–128 features.
- **With the alternatives:** route-specific pipelines would raise some routes (CSP on the motor features especially) and make differences between routes reflect pipeline quality as well as signal content.

### D15. Chance threshold: binomial, p < 0.05, per subject; checked by shuffled labels
- **Derivation:** for n trials, the threshold is the smallest k/n with P(X ≥ k) ≤ 0.05 for X ~ Binomial(n, 0.5) (Combrisson & Jerbi, 2015). n = 45 → **0.644** (29/45); n = 42 (S104) → **0.643** (27/42). p = 0.5 because classes are balanced (834 left / 828 right overall; 21–24 per class per subject).
- **Why check it:** CV test folds are not independent tests, so the binomial is only an approximation. Each decoder was also run once with shuffled labels: **5 of 296 shuffled decoders (1.7%)** crossed the threshold, so it is conservative here, not liberal.
- **Group level:** number of subjects above threshold vs Binomial(37, 0.05), and a one-sided Wilcoxon test of accuracy − 0.5 across subjects.
- **Alternative:** a per-subject permutation null (e.g. 1,000 shuffles × 10-repeat CV per route and subject).
- **With the alternative:** thresholds slightly below 0.644 would be expected, given the 1.7% false-positive rate, so a few more subjects would cross; the ordering of routes would not change.

### D16. Route definitions (channels / band / window)
- R1 ocular: Fp1, Fp2, AF7, AF8, F7, F8, FT7, FT8; raw reference; < 4 Hz; 8 × 0.5 s mean-amplitude bins over 0–4 s, baseline −0.5–0 s.
- R2 visual evoked: P7, P8, PO7, PO3, POz, PO4, PO8, O1, Oz, O2; CSD; < 20 Hz; 9 × 50 ms bins over 0.05–0.5 s, baseline −0.2–0 s.
- R3 posterior alpha: same posterior channels; CSD; 8–13 Hz log power 0.5–4 s minus baseline.
- R4 EMG: 15 peripheral channels; CSD; 30–57 + 63–78 Hz log power 0.5–4 s minus baseline (skips the 60 Hz notch).
- R5 pre-cue: all 64 channels; CSD; mu and beta log power −2 to −0.25 s.
- R5 pre-cue ocular (**added after the first A3 run, see D17**): same as R1 but −2 to −0.25 s, no baseline.
- R5 order: trial position in run + run number only.
- R6 cue timing: annotation durations per class.
- Motor reference: CSD mu and beta ERD at the 11 A2 channels.
- **Why these choices:** each route is restricted to where its mechanism must appear and, where possible, to where the motor signal cannot: the ocular route uses frontal channels below 4 Hz (mu is 8–13 Hz, central); the posterior routes use CSD so frontal EOG cannot spread back to them.
- **Limit:** the routes are not perfectly separable. Sustained gaze shifts also change posterior signals, and R3 shares the mu band. A3 shows that each route carries information, not that the routes are independent.

### D17. A test added after seeing data, stated as such
After the first A3 run, `a3_mechanisms.png` showed a ~15 µV F7−F8 class difference **before** the cue, and the sequence check showed that consecutive trials repeat the same class only 25% of the time (random order: 50%). The pre-cue test I had committed (R5, mu/beta power) cannot see a slow eye-position signal, so I added R5_precue_ocular. It is reported separately and was not part of the committed prediction.

### D18. Reference location inferred from the ocular signal
The eyes are a left/right-symmetric source, so the ocular class difference should be equally large at mirror-image sites. Under the raw reference it is **2.0× (F7/F8), 2.7× (FT7/FT8), 4.1× (T7/T8)** larger on the left; under CAR it is 0.9–1.1× (`a3_reference_asymmetry.csv`). The asymmetry is largest near the ears, which fits a reference electrode on or near the **right ear/mastoid**: it picks up part of the EOG and cancels it on right-side channels. This is the concrete form of the lateral-reference problem in D9. It is also why R1 uses the raw reference only as input to an LDA, which can rebuild any bipolar combination, and never as a measure of hemispheric asymmetry.

### D19. Below-chance order decoder is a CV artefact of a structured sequence, not a route
R5_order scores 37.5% (0/37 above threshold; shuffled 50.8%). The class sequence is strongly anti-persistent (P(same as previous) = 0.25, pooled binomial p = 3.7e-88), so labels at neighbouring positions are anti-correlated. Training folds therefore systematically mislead the position-only LDA on held-out neighbours. Position alone does not predict class (no subject above threshold). The anti-persistence does matter in another way: it turns any trace of the previous trial into information about the current one (D17).

---

## Part B — Networks (`task1/models.py`, `task1/train.py`)

### D20. Fixed subject-wise 5-fold scheme
- **Choice:** the 37 kept subjects are permuted once with the project seed and split into 5 folds of 8/8/7/7/7 (`outputs/task1/folds.csv`). Every model, seed and experiment uses the same folds, so every subject is a test subject exactly once and per-subject accuracies can be compared across models.
- **Alternatives:** leave-one-subject-out (37 folds); one fixed train/test split.
- **Why:** LOSO would multiply training cost by 7× (~1,400 runs); a single split would give B2.5 only ~7 test subjects to correlate with Part A.
- **With the alternatives:** LOSO gives ~1 more training subject per model (≈ +3% data), so accuracies would be marginally higher; a single split would make the B2.5 correlations meaningless (n ≈ 7).

### D21. Network input: 0–3.975 s, CAR, 80 Hz, per-trial scale
- **Choice:** (1) crop to the imagery period after the cue; (2) common average reference; (3) resample 160 → 80 Hz; (4) divide each trial by its own SD over channels and time.
- **Why:** (1) the pre-cue window carries the previous trial's eye position (A3 D17), which is not imagery. (2) The shared reference signal is 25–60% of the variance and differs between subjects (D9); CAR is a fixed linear map, so it uses no statistics. (3) Every route that matters (mu/beta ≤ 30 Hz, EOG, visual evoked) sits below the 40 Hz Nyquist, and 80 Hz halves the cost of ~200 training runs. (4) Amplitude differs 5–8× between subjects (D8); one scalar per trial removes that and keeps every within-trial relation.
- **Leakage check:** none of the four steps uses a statistic computed over more than the single trial being processed, so test trials cannot influence training or each other.
- **Alternatives and what would change:** CSD instead of CAR would sharpen the motor signal but needs spline fitting on a montage that B2.3 deliberately perturbs; 160 Hz would let the networks use the EMG route (A3: 53%, 6/37 subjects) and cost 2× more; per-channel z-scoring would erase between-channel power differences, i.e. the C3-vs-C4 ERD contrast itself; dataset-level z-scoring would need training-set statistics (allowed) but leave the 5–8× subject differences in place.

### D22. EEGNet hyperparameters changed for this data
The paper's lengths are in samples at 128 Hz; they are converted to seconds and back to samples at 80 Hz:

| hyperparameter | paper (128 Hz) | here (80 Hz) | why |
|---|---|---|---|
| temporal kernel length | 64 (0.5 s) | **40** (0.5 s) | same filter duration: 4 cycles of 8 Hz, resolves down to 2 Hz |
| first average pool | 4 (→ 32 Hz) | **2** (→ 40 Hz) | 32 Hz is not reachable from 80 Hz with an integer pool; rounding to 40 Hz keeps more temporal detail |
| separable kernel | 16 (0.5 s at 32 Hz) | **20** (0.5 s at 40 Hz) | same duration |
| input length | 2 s × 128 Hz | 3.975 s × 80 Hz = **318** | the imagery period here lasts 4.1 s |
| classifier input | 16 × 4 | **16 × 19** | follows from the above |

Unchanged: F1 = 8, D = 2, F2 = 16, second pool 8, dropout 0.5 (the paper's cross-subject value), max-norm 1.0 on the depthwise spatial filter and 0.25 on the classifier, ELU, batch norm. **2,642 parameters.**

### D23. Transformer design (one encoder, three tokenisers)
- **Encoder:** CLS token + 2 pre-norm blocks (d = 16, 2 heads, FF 32, dropout 0.25), classification from the CLS token. A custom block built from `nn.MultiheadAttention` replaces `nn.TransformerEncoderLayer` only so that attention weights can be read out (B2.4).
- **Time-patch tokeniser (TF-time, 6,962 params):** temporal filter bank (8 × 0.5 s) → per-filter spatial filters over all 64 channels (D = 2) → 13 non-overlapping 0.3 s patches, each pooled to 4 values per map → linear embedding + sinusoidal position.
- **Channel tokeniser (TF-chan-id 6,930 / TF-chan-noid 5,906 params):** the same 8-filter temporal bank applied to every electrode separately (shared weights) → 8 × 0.5 s segments → linear embedding; plus a learned 64 × 16 electrode-identity embedding in the "id" variant.
- **Budget:** 3 × 2,642 = 7,926; all three variants are within it.
- **Why:** the front ends deliberately reuse EEGNet's ideas (band filters of 0.5 s, depthwise spatial filters), so that comparisons with EEGNet isolate what attention adds and comparisons between tokenisers isolate the tokeniser. d = 16 is what the parameter budget allows with two layers.
- **Alternative:** a larger patch embedding straight from raw samples (64 ch × patch → d) would need 64 × 24 × 16 ≈ 25k parameters per embedding alone, over budget.

### D24. Training: fixed length, no early stopping, no tuning on held-out subjects
- **Choice:** AdamW (lr 1e-3, weight decay 1e-2), batch 64 sampled with replacement, 5% warm-up then cosine decay; length = 25 passes over the training set, clipped to 200–500 steps; no early stopping, no validation set. All hyperparameters were fixed before any test accuracy was seen. The only pre-run check was the **training** loss on fold 0's training subjects (EEGNet 0.36, TF-time 0.43, TF-chan-id ≈ 0.45 at 500 steps, flat over the last 100–200 steps).
- **Alternatives:** early stopping on held-out subjects; tuning hyperparameters on a validation fold.
- **Why:** every subject is a test subject in some fold, so any tuning on held-out subjects leaks into some fold's test set; the brief forbids early stopping on test subjects.
- **With the alternatives:** done properly (inner validation subjects inside each training pool) it would cost one fewer training subject and probably gain 1–2 points; done improperly it would inflate accuracy.

### D25. Seeds
- Run seed = project seed + 100 × seed index + fold; seed index ∈ {0, 1, 2}. The seed sets weight initialisation, batch sampling and, in B2.1, which training subjects are drawn. Training subsets are nested (5 ⊂ 10 ⊂ 20 ⊂ all) per (fold, seed), and EEGNet and the transformer get the same subjects.

### D26. Displacement: rigid cap rotation, 0–15°, both axes and both signs
- **Choice:** rotate all 64 positions together on the fitted head sphere by 2.5, 5, 10, 15° (3.9, 7.9, 15.7, 23.6 mm of arc at Cz for a 9 cm radius), towards each ear (lateral) and towards nose/back (AP); report the two signs of each axis averaged, plus each sign separately in `b2_3_displacement.csv`.
- **Size justification:** across 20 participants, electrode placement varied with SD 3.9 mm (left–right), 5.6 mm (front–back) and 7.2 mm (vertical) ("Variability of EEG electrode positions and their underlying brain regions", *Brain and Behavior*, 2022). 2.5–5° is routine variation; 10–15° (≈ 2–4 SD) is a clearly misplaced cap.
- **Alternatives:** independent per-electrode jitter; a translation of the whole cap.
- **Why:** a cap moves as one piece, so a rigid rotation is the physically plausible error; independent jitter would mostly add spatial noise.
- **With the alternatives:** jitter of the same RMS size would hurt less, because it doesn't shift the left/right balance coherently.

### D27. Spherical-spline regularisation λ = 1e-9 (not MNE's 1e-5)
- **Found:** with MNE's default λ = 1e-5, interpolating the 64 channels **onto themselves** (0°) gives max |M − I| = 0.67; each channel keeps only 39% of its own signal. Every "displacement" would then include a large smoothing that has nothing to do with misplacement.
- **Choice:** λ = 1e-9: 0° map is the identity (max |M − I| = 0.004); at 10° the worst-case channel gain is 1.86 (edge electrodes rotated past the cap edge) and total signal power is preserved within 3%; a smooth test field is reproduced to 2e-4.
- **Implementation:** Perrin et al. (1989), m = 4, 50 Legendre terms; checked against `mne.channels.interpolation._make_interpolation_matrix` (identical at equal λ).

### D28. Attribution: token-level gradient × input; deletion by removing tokens
- **Choice:** score_t = |Σ_d e_t,d · ∂(logit_pred − logit_other)/∂e_t,d| on the token embeddings (incl. positional encoding), from all 15 TF-time models on their own test folds; deletion removes tokens from the sequence (the encoder handles any length), ranks recomputed per trial; random baseline = 5 random orders per trial.
- **Alternatives:** integrated gradients; replacing tokens with zeros or a mean token.
- **Why:** gradient × input is the simplest gradient-based token score and has the same token granularity as attention; removal avoids feeding the model an out-of-distribution placeholder.
- **Known mismatch (see B2.4 outcome):** gradient × input estimates the effect of **zeroing** a token linearly, while the test **removes** it; the two differ through the softmax renormalisation and LayerNorm.
- **Saliency maps for Part A:** input-level |gradient × input| per electrode, normalised per trial; summarised as the share on the 21 ocular sites and on the 11 A2 sensorimotor channels relative to an even spread (`b2_4_saliency_shares.csv`). The A3 single-channel maps are not used for this, because they were computed under the raw reference, which skews the ocular map to the left (D18).

### D29. Removing the dominant confound (ocular): drop frontal sites + CAR over the rest + 8–30 Hz
- **Choice:** set the 21 Fp/AF/F-row sites plus FT7/FT8/T9/T10 to zero, re-reference to the average of the remaining 43 channels, band-pass 8–30 Hz on the whole −2 to 4 s trial, then the usual crop/resample/scale (`train.preprocess_variant`). Same folds, seeds and training length as the main runs.
- **Alternatives:** (a) regress EOG proxies (F7−F8, Fp mean) out of every channel; (b) ICA; (c) only one of channel removal or band-pass.
- **Why:** there are no EOG channels, so regression would need coefficients estimated from the test subject's own data (a normalisation statistic, forbidden) or transferred from training subjects (poor fit, since the reference geometry differs, D18). ICA needs per-subject fits on test data for the same reason. Removing sites alone leaves the eye signal volume-conducted to central channels below 4 Hz; band-pass alone leaves saccadic spike potentials and the frontal 8–30 Hz content. Together they close the route.
- **Verified on the network inputs** (within-subject LDA, `b2_5_removal_check.csv`): slow time-locked amplitude (all channels) 73.1% → 49.3%; posterior evoked 61.3% → 49.1%; sensorimotor mu/beta power 61.7% → 62.1%. The method removes what it targets and leaves the motor features untouched.
- **What it cannot remove:** small covert hand movements during "imagery" produce sensorimotor ERD indistinguishable from imagery without EMG; the networks never see > 40 Hz, so the A3 EMG route is closed to them but not ruled out as a behaviour.

### D30. B2.6 intervention: per-channel pre-cue normalisation inside each trial
- **Choice:** in the ocular-removed input, divide each channel by its own 8–30 Hz RMS over −1.5 to −0.5 s of the same trial (`no_ocular_erdnorm`), so the network sees ERD (log power change from baseline), not absolute power. Rationale and maths in PREDICTIONS.md (B2.6).
- **Alternatives:** Euclidean alignment / per-subject covariance whitening; a sweep of architectures or band limits.
- **Why:** alignment methods need statistics over the test subject's trials (forbidden here); a sweep would not test a mechanism. This change targets the one quantity A2 identified as the motor signal.
