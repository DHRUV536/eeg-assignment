# Predictions

Each prediction is committed before the commit that contains its result. The outcome is
appended afterwards under **Outcome**, without editing the prediction.

Disclosure: this repository's history (commits 662da55–86b15d1) contains an earlier version of
this assignment with different specifications. The predictions below were written knowing that
history exists; they are made for the current pipeline (37 kept subjects, CSD, the decoders
specified here), whose results had not been computed when each prediction was committed.

---

## P1 — A3 confound check (committed before any A3 code was run)

**How trials were cued** (PhysioNet EEGMMIDB page, protocol for runs 4/8/12): a target appears
on the left or right side of the screen; the subject imagines opening/closing the corresponding
fist until the target disappears (~4.1 s), then relaxes. The page mentions no fixation
instruction. So the class label is perfectly confounded with *where on the screen* a stimulus is.

**Route 1 — eye movements towards the target.** The corneo-retinal dipole makes a horizontal
saccade a large, polarity-reversed potential at lateral frontal electrodes; there are no EOG
channels, so F7−F8, AF7−AF8 and Fp1−Fp2 serve as horizontal-EOG proxies (bipolar, hence
reference-free). Decoder: those 3 derivations, 0.1–8 Hz, mean amplitude in 250 ms bins from
0 to 3.5 s after the cue, baseline −0.2–0 s (14 bins × 3 = 42 features). The window covers the
saccade (~0.2–0.6 s) and any sustained gaze offset while the target stays on screen.

**Route 2 — lateralised visual response to the target's hemifield.** A target in the left
hemifield is processed by the right occipital cortex, giving a lateralised evoked response and,
while attention stays there, posterior alpha lateralisation. Two decoders, both restricted to
occipito-parietal channels:
- *evoked*: PO7−PO8, PO3−PO4, O1−O2, 1–40 Hz, mean amplitude in 50 ms bins 0.05–0.50 s
  (9 × 3 = 27 features);
- *alpha*: CSD, log(8–13 Hz power 0.5–3.5 s / baseline −1.5 to −0.5 s) at PO7, PO3, O1, PO8,
  PO4, O2 (6 features). CSD so that sensorimotor mu does not volume-conduct into it.

**Positive control — motor route.** CSD, C3/C4/C1/C2/C5/C6/FC3/FC4/CP3/CP4, log mu and beta
power ratio 0.5–3.5 s vs baseline (20 features).

All decoders: within-subject, standardise + shrinkage LDA, leave-one-run-out CV (3 folds; trials
within a run share slow drifts, so random k-fold would leak). Chance threshold per subject:
smallest k with P(X ≥ k | n, 0.5) ≤ 0.05 under a binomial null, threshold = k/n (≈ 64% for
n = 45). Group level: count of subjects above threshold vs the ~1.9 expected by chance, and a
one-sided Wilcoxon of accuracies vs 0.5.

**Prediction.**
1. *Ocular* will be the most decodable non-motor route: above threshold in roughly 8–15 of 37
   subjects, group mean ≈ 57–60%, clearly above chance at group level. Reason: nothing in the
   protocol stops subjects from looking at a target that stays on screen for 4 s, and frontal
   bipolar derivations capture horizontal EOG with high SNR.
2. *Visual evoked*: weak; group mean ≈ 52–54%, ≤ 4 subjects above threshold. A lateralised VEP
   is clear in averages but small in single trials of 45-trial subjects.
3. *Visual alpha*: near chance; ≤ 3 subjects above threshold (posterior alpha lateralisation is
   an attention effect that is unreliable without an explicit covert-attention task).
4. *Motor control*: group mean ≈ 57–60%, ~8–10 subjects above threshold (A2 found 10 with
   uncorrected mu p < .05).
5. Ocular and motor accuracies will be only weakly correlated across subjects (|r| < 0.3), i.e.
   they are different subjects' "routes", and at least one A2 *absent* subject will be strongly
   ocular-decodable.

### Outcome P1 (from `outputs/task1/a3_summary.csv`, `a3_decoders.csv`, `a3_evidence.png`)

| route | predicted | observed (mean acc; subjects ≥ threshold) |
|---|---|---|
| ocular | 57–60%; 8–15/37 | **76.0%; 31/37** (Wilcoxon p < .001) |
| visual evoked | 52–54%; ≤ 4/37 | **64.5%; 20/37** (p < .001) |
| visual alpha | near chance; ≤ 3/37 | 51.0%; 2/37 (p = .39) ✓ |
| motor (control) | 57–60%; 8–10/37 | 52.7%; 8/37 (p = .13) — count ✓, mean lower |
| ocular vs motor, across subjects | \|ρ\| < 0.3 | ρ = −0.03 ✓ |
| ≥ 1 A2-absent subject ocular-decodable | yes | 26 of the 32 A2-absent subjects ✓ |

1. **Ocular — direction right, size badly underestimated.** The grand-average F7−F8 trace shows a
   ±45 µV step of opposite polarity for left vs right targets starting ~0.2 s after the cue,
   i.e. a saccade *towards* the target (a leftward saccade brings the positive cornea towards F7),
   in nearly every subject. Looking at the target was the norm, not the exception. The non-motor
   route decodes left/right far better than the motor route itself.
2. **Visual evoked — wrong.** It decoded well, but its grand-average PO7−PO8 peak sits at ~0.2 s,
   the saccade onset, and left/right traces already differ before the cue (the zero-phase
   1–40 Hz filter spreads the saccade potential backwards in time). **Post-hoc check** (added after
   seeing these results, not predicted): restricted to 0.05–0.15 s, before saccades, on
   0.1–40 Hz data, visual decoding falls to 51.8% (5/37, Wilcoxon p = .22) and ocular to 52.8%
   (4/37). So the "visual" route is mostly the same eye movement seen from occipital sites, not
   an independent lateralised visual response.
3. **Motor control** was weaker than predicted on average: with ~30 training trials, 20 power
   features are hard to fit, and only a minority of subjects lateralise (A2: 5/37 after FDR).

---

## P2 — B2.1 data scaling (committed before any network was trained)

Setup: test on 10 fixed subjects; train on nested random subsets of 5, 20 and all 23 training
subjects; one seed; EEGNet vs the time-patch transformer (D19–D27).

Prediction, using what A3 showed: the left/right **eye movement** is large (±45 µV at F7−F8),
has the same polarity in nearly every subject, and is inside the 0–4 s input. Both networks
will learn it, so cross-subject accuracy will sit well above the motor-control level (52.7%
within subject) and close to the within-subject ocular decoder (76%).

| | 5 subjects | 20 | all 23 |
|---|---|---|---|
| EEGNet | ~62% | ~70% | ~71% |
| Transformer (time-patch) | ~55% | ~66% | ~68% |

- EEGNet ≥ transformer at every size, with the largest gap at 5 subjects and **no crossover**.
  EEGNet's structure (temporal filters, then one spatial filter per temporal filter) is a
  strong prior: a single depthwise spatial filter can form an F7−F8-like difference, and the
  same structure is a learnable filter-bank CSP for ERD. The transformer must learn its spatial
  projection, temporal filters and attention from fewer constraints, so it needs more data.
- 20 → 23 subjects changes little (+3 subjects): both curves flatten.
- With ~445 test trials, the pooled-accuracy SE is ≈ 2.4 points; with one seed, differences under
  ~5 points will not be called real.

## P3 — B2.2 tokenisation (committed before any network was trained)

All 23 training subjects, same seed.

1. **Time-patch ≈ channel + identity** (within ~4 points), both ≈ 65–70%: both can tell
   electrodes apart (the time-patch spatial projection has one weight per electrode; the
   identity embedding labels each channel token).
2. **Channel tokens without identity ≈ chance (48–55%).** With shared token encoders,
   attention and mean pooling, the model is exactly invariant to permuting electrodes (verified
   in code): it sees an unordered *set* of 64 channel time series. It therefore cannot represent
   anything that depends on *which* electrode carries a signal — no hemisphere contrast, no
   "C3 vs C4", no "F7 vs F8". Both left/right signals in this data are mirror-symmetric: ERD at C4
   vs C3, and the EOG step positive at F7 / negative at F8 vs the reverse. The set of channel
   series is nearly the same for left and right trials, so only incidental asymmetries
   (e.g. leftward vs rightward saccades of different size) are left to use. For telling left
   from right this limitation is fatal.

### Outcome P2 (from `outputs/task1/b2_runs.csv`, `b2_scaling_tokenisation.png`; 448 test trials, chance threshold 53.8%)

| | 5 subjects | 20 | all 23 |
|---|---|---|---|
| EEGNet — predicted / observed | ~62% / **76.8%** | ~70% / **77.7%** | ~71% / **78.6%** |
| Transformer (time-patch) — predicted / observed | ~55% / **49.3%** | ~66% / **63.6%** | ~68% / **64.3%** |

- ✓ EEGNet above the transformer at every size, largest gap at 5 subjects (27 points), no crossover.
- ✗ **EEGNet's curve is flat**, not rising: 5 training subjects already give 76.8% (per test
  subject 0.60–0.98). I expected the cross-subject problem to need many subjects. It does not,
  because the feature EEGNet can use is one large, same-polarity-in-everyone eye-movement
  potential: one depthwise spatial filter (≈ F7 − F8) after a low-frequency temporal filter
  captures it, and ~225 trials are enough to fit that. Its accuracy matches the within-subject
  ocular decoder (76%), not the motor control (53%).
- The transformer is at chance with 5 subjects (49.3%) and only reaches ~64% with 20+: it has to
  learn its spatial projection, temporal filters and attention without EEGNet's built-in
  "temporal filter → per-filter spatial filter" structure.

### Outcome P3

| tokenisation (all 23 subjects) | predicted | stop on val loss (pre-registered) | stop on val accuracy (post-hoc, D28) |
|---|---|---|---|
| time-patch | 65–70% | 64.3% | 65.6% |
| channel + identity | ≈ time-patch | **56.5%** (best epoch = 1) | 69.9% |
| channel, no identity | 48–55% (chance) | **57.4%** (p = .001) | 57.4% |

1. **Time-patch vs channel + identity — ✗ under the pre-registered rule, ✓ after the post-hoc check.**
   The channel + identity model's validation loss rose from the first epoch while its validation
   accuracy kept improving (over-confident predictions on the 4 validation subjects), so the
   loss rule kept its epoch-1 weights. Stopping on validation accuracy instead, the two
   identity-aware tokenisations are within 4.3 points (65.6% vs 69.9%), as predicted.
2. **No identity — ✗, above chance (57.4%, 257/448, one test subject at 82%).** The model is exactly
   permutation-invariant (checked), so the prediction's logic holds only if the left/right
   difference is mirror-symmetric across electrodes. It is not. Post-hoc check
   (`b2_2_common_mode.csv`, D29): during the saccade (0.2–1 s), left-minus-right is +11.7 µV at F7
   but only −7.7 µV at F8, 52/64 channels shift the same way, and the **mean over all 64
   channels** differs by +2.4 µV between left and right trials (32/37 subjects, p = 6e-5). An
   order-blind model can read such a global shift. The asymmetry is what you would expect if the
   undocumented recording reference sits off the midline (a horizontal eye movement then moves
   the reference itself): the same unknown reference A2 had to work around. What the model
   cannot represent is still any C3-vs-C4 (or F7-vs-F8) *contrast*, so it cannot use lateralised
   ERD at all; it can only use the reference-induced common-mode part of the eye movement.

---

## P4 — B2.3 what the model relies on (committed before any attribution was computed)

Model: the best transformer by validation accuracy (D28): `tf-time_nall_stopacc`
(time-patch tokens, test 65.6%). EEGNet (`eegnet_nall`, 78.6%) gets the same channel-level
analyses for comparison.

Analyses: Integrated Gradients (baseline = all-zero input = training mean after z-scoring,
32 steps, target = predicted class) on every test trial.
(i) token deletion: per trial, replace the input samples of the k most-attributed time tokens
(250 ms each) with the training mean vs k random tokens (10 random draws), k = 0…8;
(ii) attribution share per channel group (frontal/ocular 17 ch: Fp*, AF*, F*; sensorimotor 21:
FC*, C*, CP*; parieto-occipital 18: P*, PO*, O*, Iz; rest 8: FT/T/TP) and per time token;
(iii) group deletion: zero each channel group vs random 17-channel sets;
(iv) per test subject (n = 10), Spearman correlation of model accuracy with the A2 mu LI and the
A3 ocular, visual-evoked and motor decoder accuracies.

**Prediction.**
1. Deletion is faithful: removing the top-2 attributed tokens drops accuracy to ≈ 55%, removing 2
   random tokens to ≈ 63% (from 65.6%).
2. Attribution peaks in 0.25–1.5 s (saccade towards the target and the return), not evenly over
   the 4 s imagery period.
3. Per channel, frontal/ocular channels carry > 2× the attribution of sensorimotor channels, for
   both models.
4. Zeroing frontal/ocular channels drops both models to ≈ 52–58%; zeroing sensorimotor channels
   costs ≤ 3 points; random 17-channel sets cost little.
5. Per-subject accuracy correlates positively with A3 ocular accuracy (ρ > 0.5) and not with the
   mu LI (|ρ| < 0.3). Caveat fixed in advance: all 10 test subjects are A2-absent (LI −1.2 to
   +2.1 dB), so the LI correlation has little range and n = 10 gives wide uncertainty.
6. Conclusion expected: the models read the **A3 ocular confound**, with at most a small
   sensorimotor contribution.

### Outcome P4 (from `outputs/task1/b2_3_*.csv`, `b2_3_reliance.png`)

| | predicted | observed |
|---|---|---|
| 1. delete top-2 vs random-2 tokens (transformer, 65.6%) | ≈ 55% vs ≈ 63% | **66.1% vs 64.9%**; top-8 vs random-8: 59.2% vs 61.5% |
| 2. attribution in 0.25–1.5 s (uniform = 31%) | concentrated | transformer 41%, EEGNet 51% (peak 0.25–1.0 s) |
| 3. frontal/ocular per channel vs sensorimotor | > 2× | transformer 0.83× vs 1.14× of uniform; EEGNet 1.10× vs 0.99× (temporal 1.31×) |
| 4. zero frontal/ocular (17 ch) | 52–58% both | transformer **53.8%** ✓; EEGNet **77.7%** (−0.9) ✗ |
| 4. zero sensorimotor (21 ch) | ≤ 3-point cost | transformer +1.6, EEGNet −2.0 ✓ |
| 5. ρ(model acc, A3 ocular), n = 10 | > 0.5 | transformer **+0.64 (p = .045)** ✓; EEGNet +0.45 (p = .19) |
| 5. ρ(model acc, A2 mu LI) | \|ρ\| < 0.3 | transformer +0.04 ✓; EEGNet +0.57 (p = .09) ✗ |

- **Deletion was only weakly faithful (✗).** Removing the 1–2 highest-IG time tokens does not hurt
  more than removing random ones; only from 4 tokens on does it hurt slightly more. Two reasons:
  the 41-sample temporal filters straddle the 40-sample token boundaries, so information about
  a deleted token survives in its neighbours; and the evidence (a saccade lasting ~1 s, then a
  sustained offset) is spread over several tokens, so no single token is decisive.
- **Region shares were wrong (✗).** I expected attribution to sit on frontal channels. It is spread
  out, because the eye-movement potential is not frontal-only: it is as large at FT7/T7 as at
  F7 (A3/B2.2 maps), and through the asymmetric reference it reaches every channel.
- **EEGNet does not need the frontal channels (✗)**: it reads the same signal from temporal and
  other channels (its top channels: C4, FT7, F8, AF7, FC5, F7). Zeroing one region at a time
  cannot remove a volume-conducted signal, so a post-hoc test was added (D30): regress the three
  horizontal-EOG proxies out of every channel (coefficients fit on training subjects only).
  **Transformer: 65.6% → 52.7% (chance). EEGNet: 78.6% → 65.0%.**
- Cross-subject versions of the A3 decoders on the same split: the 3-pair HEOG LDA scores **79.7%**
  on the test subjects (more than EEGNet), visual-evoked 62.1%, the motor (mu/beta CSD) decoder
  48.0%.

**Conclusion.** The best transformer reads the **A3 ocular confound** and essentially nothing
else: its accuracy goes to chance when the frontal channels or the HEOG component are removed,
and across test subjects it tracks the ocular decoder, not the mu LI. EEGNet reads the ocular
confound for about half of its above-chance accuracy (13.6 of 28.6 points removed by a linear
HEOG regression); the remaining 65% is not explained by this analysis: it could be ocular
signal the linear regression misses (e.g. the saccadic spike potential, non-linear or delayed
components) or genuine sensorimotor activity (C4 and FC5 are among its top channels). The
per-subject LI correlation cannot settle it: n = 10, all A2-absent, ρ = +0.13 after regression.
So: transformer → confound; EEGNet → confound plus an unresolved remainder.
