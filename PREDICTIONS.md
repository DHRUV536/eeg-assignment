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
