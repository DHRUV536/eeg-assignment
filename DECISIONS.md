# Decisions

Format: **choice** / alternative / why / what would change with the alternative.

## A1 — data audit (`task1/data_audit.py`)

**D1. Runs 4, 8, 12 only.** / Add executed runs 3, 7, 11. / The assignment asks for imagined movement; in runs 4/8/12 T1 = left fist, T2 = right fist (verified: PhysioNet EEGMMIDB page, "Experimental Protocol"; event codes confirmed in the EDF annotations). / Executed runs would add real movement, EMG and larger ERD, inflating every Part A/B number relative to imagery.

**D2. Exclude S088, S092, S100.** / Resample to 160 Hz and keep. / Our inspection of the EDF headers and annotations: all three are 128 Hz with a different cue timing (S088/S092: 5.12 s cues, 1.38 s rest; S100: 5.12 s cues and rests, only 12 cues per run). With 1.38 s rest there is no pre-cue baseline, and resampling fixes the rate, not the protocol. / Keeping them would add ~150 trials whose timing (and therefore ERD window) differs from everyone else's; S100 would add a subject with 18 trials per class.

**D3. S104: keep the complete trials of truncated run 8.** / Drop run 8 or the subject. / The run ends at 106 s; trials whose analysis window fits are unaffected. / Dropping the subject loses ~42 good trials for no gain.

**D4. Interpolate single dead/noisy channels (S079 Fp1/P1, S086 Fp1, S097 AF7, S105 FC5/FC1).** Detection: channel std < 0.2x or > 5x the run median. / Exclude the subject, or drop the channel everywhere. / A fixed 64-channel layout is needed by CSD (A2) and the networks (B); one bad electrode is not a reason to lose a subject. / Caveats: Fp1/AF7 feed the A3 ocular decoder; FC5/FC1 are C3 neighbours, so S105's C3 Laplacian partly uses interpolated data.

**D5. Annotation lengths of 4.0/4.15 s (S072-74, 76, 89, 102) treated as benign; in S072/73/74/76/102 the last cue of some runs then ends a few samples before the file does, so its −2..4 s window does not fit and that trial is dropped (13 trials in total; 827 left / 822 right kept).** / Shorten the window for those subjects, or pad. / A ≤0.1 s rounding of cue length; one trial per run is not worth a subject-specific window. The audit's fit check counts the last sample the same way `mne.Epochs` does (fixed after Part B's loader found 13 fewer trials than the first audit reported). / A shorter window everywhere would recover 13 trials and lose 0.1 s of imagery from all 1649.

**D6. Trial window −2.0 to 4.0 s.** / −1.5 to 4.5 s. / The last cue of each run is 4.1 s from the end of the file, so tmax > 4.0 drops 3 trials per subject; the extra 0.5 s before the baseline absorbs wavelet edge effects.

## A2 — ERD lateralisation (`task1/a2_erd.py`)

**D7. Surface Laplacian (MNE CSD) as the spatial filter.** / As-recorded reference; common average reference. / The recording reference is undocumented (dataset page). Anything at the reference appears in every channel, so a C3−C4 difference mixes in whatever the reference picks up, and if the reference lies over or near one sensorimotor cortex it can create or cancel lateralisation. CSD is reference-free and local. / Measured (`a2_reference_check.csv`): mu-lateralised ERD present in 1/37 subjects as recorded, 3/37 with CAR, 5/37 with CSD; 4 labels change between as-recorded and CSD.

**D8. Baseline −1.5 to −0.5 s, per trial.** / A baseline averaged over all trials; a baseline ending at 0 s. / It lies inside the 4.1 s T0 rest, ends 0.5 s before the cue (no anticipation, no wavelet smearing of the cue response), and starts ~2.6 s after the previous imagery ended (post-imagery beta rebound mostly over). A per-trial baseline absorbs slow drifts within a run. / A baseline common to all trials cancels exactly in the left-minus-right LI, so only noise would change.

**D9. Average power before taking the log.** / dB per time–frequency pixel, then average. / Mean of logs < log of mean: the per-pixel version put the baseline at about −2.3 dB instead of 0 and made single-subject maps noisy (first run of this code). / Biased ERD magnitudes; LI almost unchanged because the bias is similar at C3 and C4.

**D10. Imagery window 0.5–3.5 s; mu 8–13 Hz, beta 13–30 Hz; Morlet with freq/2 cycles.** / Window from 0 s. / The first 0.5 s contains the visual evoked response to the cue (visible as a broadband peak at ~0.2 s in `a2_summary.png`); ending at 3.5 s avoids the epoch edge.

**D11. LI = mean(C3−C4 | left) − mean(C3−C4 | right) of mu (8–13 Hz) ERD in dB; positive = expected (contralateral ERD).** One-sided label-permutation test (5000 permutations, seeded per subject); BH-FDR across the 37 subjects; *present* if q < 0.05. Beta (13–30 Hz) is computed the same way as a supplementary column but does not affect the label. / Two-sided test; uncorrected p; a mu-or-beta rule. / Physiology predicts the direction; the assignment defines the label on mu; FDR because ~2 of 37 subjects would pass p < .05 by chance. / Uncorrected p would label 10 subjects present instead of 5; adding beta (either band) would add one more (S108).

**D12. Example subjects chosen by rule.** Present: strongest and weakest significant subject by mu q; absent: closest to significance and clearest null (ties broken by mu effect size). / Pick by eye. / Avoids showing only the best case.

## A3 — confound check (`task1/a3_confounds.py`)

**D13. The two routes: eye movements towards the target, and the lateralised visual response to the target's hemifield.** / Within-run drift, trial order, muscle activity. / Both follow directly from the cue (a target shown on the left or right of the screen for ~4 s, PhysioNet protocol page, no fixation instruction mentioned), and both produce left/right-specific signals. Drift is handled by leave-one-run-out CV rather than tested as a route; muscle activity during imagery would be motor-related, not a cue confound.

**D14. Horizontal-EOG proxies F7−F8, AF7−AF8, Fp1−Fp2 at 0.1–8 Hz.** / ICA eye components; 1–40 Hz data. / No EOG channels exist; bipolar lateral frontal pairs reverse polarity for horizontal eye movements and are reference-free. 0.1 Hz keeps the sustained gaze offset that a 1 Hz high-pass would remove. / ICA adds a component-selection choice per subject; 1–40 Hz would keep the saccade transient but lose the sustained part (lower accuracy expected).

**D15. Leave-one-run-out CV (3 folds) within subject.** / Stratified random 5-fold. / Trials in the same run share slow drifts (impedance, alertness); random folds put neighbouring trials in train and test and inflate accuracy. / Random k-fold would likely give higher accuracies for every route.

**D16. Chance threshold: binomial, smallest k with P(X ≥ k | n, 0.5) ≤ .05 (Combrisson & Jerbi 2015), per subject n (≈64% for n = 45).** Group level: count of subjects above threshold vs the 1.85 expected by chance (binomial test), and Wilcoxon of accuracies vs 0.5. / Label-permutation threshold. / Closed form, per subject, and what the assignment asks for. CV predictions are not perfectly independent, so this threshold is slightly lenient; the group tests do not rely on it. / Permutation thresholds would be similar or a few points higher.

**D17. Standardise + shrinkage LDA (Ledoit–Wolf).** / Logistic regression with tuned C. / ~30 training trials and up to 42 features: shrinkage regularises the covariance without a hyperparameter search that would need nested CV. / Similar accuracies, more tuning.

**D18. Post-hoc pre-saccade window (0.05–0.15 s, 0.1–40 Hz).** Added after seeing the results (marked `posthoc` in the CSVs): the visual-evoked peak coincided with saccade onset and the zero-phase FIR smeared the saccade potential into the baseline. Ending the window before saccades separates an early visual response from the eye movement. Result: both routes ≈ chance there, so the visual decoder's 64.5% is attributed to eye movements.
