# PREDICTIONS

Each prediction is committed before the commit that contains its result (see `git log`).
What I had seen when writing each one is stated, so it is clear what the prediction could and could not draw on.

---

## A3 — Which non-motor route will be decodable?

**Written after A1 and A2, before any A3 decoder was run.** Already known at this point: Fp/AF channels exceed 1 mV in 24/40 subjects (A1), and the CSD lateralisation analysis finds lateralised mu/beta ERD in 18/37 subjects (A2). No left-vs-right comparison of any non-motor signal had been made.

**How trials were cued** (PhysioNet eegmmidb page): "a visual target appeared on either the left or right side of the screen"; the subject imagined opening and closing the corresponding fist "until the target disappeared", then relaxed. The target stays on screen for the whole 4.1 s imagery period. No fixation instruction is documented.

**Routes by which left and right trials can differ without motor imagery:**

| # | Route | Mechanism | Where/when it should appear |
|---|---|---|---|
| R1 | Eye movements | gaze moves toward the target side → horizontal EOG, opposite polarity at left vs right lateral-frontal sites | F7/F8, AF7/AF8, FT7/FT8, Fp1/Fp2; < 4 Hz; from ~0.2 s after the cue |
| R2 | Visual evoked response | a stimulus in the left hemifield drives the right visual cortex first (P1/N1 larger contralaterally) | O1/O2, PO7/PO8, P7/P8; 0.05–0.5 s |
| R3 | Visuospatial attention | attending one side lowers posterior alpha over the opposite hemisphere; same band and same hemispheric sign as motor mu ERD | posterior channels; 8–13 Hz; sustained |
| R4 | Muscle (EMG) | tension or small real movements (hand, arm, neck, jaw), and saccadic spike potentials from the eye muscles | peripheral channels (temporal, frontal, occipital edge); 30–78 Hz |
| R5 | Trial order and carry-over | if the left/right sequence is not random, or the previous trial leaves a trace, the class is predictable before the cue | pre-cue window (−2 to −0.25 s); trial position in run |
| R6 | Cue timing | if left and right cues had different durations or timing, the window itself would differ | annotation durations |

**Prediction:**
- **R1 (ocular): decodable.** It is the strongest non-motor route. With a peripheral target and no fixation instruction, most subjects will look at it at least sometimes. Expect a group-mean accuracy of about 60%, above the single-subject chance threshold in roughly a third of subjects.
- **R2 (visual evoked): weakly decodable at group level** (mean ≈ 54–55%), but above the single-subject threshold in only a few subjects, because 45 trials give a low SNR for an evoked lateralisation.
- **R3 (posterior alpha): mostly not decodable** (≈ 52%). If it is present, it matters more than its size suggests, because it lives in the mu band with the same hemispheric sign, so a network looking at "mu lateralisation" cannot tell it apart without spatial precision.
- **R4 (EMG): not decodable** at group level (≈ 51–52%), with possibly 1–3 individual subjects above threshold.
- **R5 (order / pre-cue): not decodable** (≈ 50%): PhysioNet BCI2000 runs are expected to randomise targets.
- **R6 (cue timing): no difference**; all cue durations were 4.1 s in A1 except S104's truncated trial.
- **Reference point:** a sensorimotor decoder (CSD mu/beta at C3/C4 and neighbours) should reach ≈ 60–62% on average, and should be above threshold mainly in subjects A2 labelled "present".

**Main reason:** the target is visual, peripheral and present for the whole trial, so the eyes are the most direct non-motor path from the cue to the recording, and the lateral-frontal channels pick it up at amplitudes 10–100× larger than mu ERD.

### A3 outcome (added with the results commit; the prediction above is unchanged)
| Route | Predicted | Observed mean acc. (subjects above 0.644) | Verdict |
|---|---|---|---|
| R1 ocular | decodable, ≈ 60%, ~⅓ of subjects | **71.7%** (25/37) | right direction, **much larger** than predicted |
| R2 visual evoked | ≈ 54–55% group, few subjects | **60.7%** (13/37) | larger than predicted |
| R3 posterior alpha | ≈ 52%, mostly not | 51.2% (1/37) | as predicted |
| R4 EMG | ≈ 51–52%, 1–3 subjects | 53.4% (6/37) | slightly larger; 6 subjects, S108 at 84% |
| R5 pre-cue power | ≈ 50% | 48.1% (1/37) | as predicted |
| R5 sequence | random | **anti-persistent**: P(same as previous) = 0.25 | **wrong** |
| R5 pre-cue ocular (added, D17) | — | 57.0% (7/37) | not predicted |
| R6 cue timing | no difference | 0.000 s | as predicted |
| motor reference | ≈ 60–62% | **55.3%** (7/37) | **lower** than predicted |

**Where the reasoning went wrong.** I expected ocular > motor but underestimated by how much: the restricted ocular decoder beats the restricted motor decoder by 16 points on average and in 34/37 subjects. I assumed BCI2000 randomises the cue sequence; it alternates far more than chance, which turns the previous trial's eye position into a pre-cue predictor of the next class.

---

## B2.1–B2.3 (written after A3, before any network was evaluated on a test subject)
Already known: A3 (ocular route 72% within-subject, motor route 55%), training-loss curves on fold 0's training subjects. No test-set accuracy of any network had been computed.

### B2.1 Data scaling (5, 10, 20, all ≈ 29–30 training subjects; 3 seeds; 5 folds)
- **Both models improve with more subjects; no crossover within this range.** EEGNet stays ahead at every size, with the gap shrinking from ≈ 5 points at 5 subjects to ≈ 2–3 points at all subjects.
- **Levels:** EEGNet ≈ 60% (5 subjects) → ≈ 68–72% (all); TF-time ≈ 55% → ≈ 65–68%. Seed-to-seed spread ±2–3 points at small N, ±1–2 at all.
- **Why:** EEGNet hard-codes what EEG decoding needs (a temporal filter bank and fixed spatial filters, then a separable conv) with 2.6k parameters. TF-time has the same front end but replaces the separable conv with attention over 13 time tokens (7k parameters), which has to learn from data which time segments matter. The transformer's weaker prior pays off only with far more data than 30 subjects × 45 trials. Accuracy will be high mostly because the dominant signal (horizontal EOG, F7/F8 polarity flip) has the same spatial pattern in every subject, so it transfers across subjects much better than mu ERD.

### B2.2 Tokenisation (all training subjects)
- **TF-chan-noid ≈ 50–53% (near chance).** Without an identity embedding, channel tokens are an unordered set; the CLS readout is invariant to permuting electrodes. Left vs right differs by a **mirror** of the same spatial pattern: EOG is +a at F7 and −a at F8 for one direction and the reverse for the other; ERD is low power at C4 for left and at C3 for right. Both give the **same multiset of channel signals**, so they cannot be told apart. Any remaining accuracy has to come from left/right-asymmetric content (anatomy, cap placement, channel-specific noise).
- **TF-chan-id ≈ 62–66%, below TF-time.** Identity restores "where", but spatial combinations (e.g. F7 − F8) have to be built through attention between tokens after each channel was processed alone, which is harder at d = 16 than TF-time's direct spatial filters.
- **TF-time ≈ the B2.1 transformer result (~65–68%).**
- **Does it matter for this task?** Yes. The whole task is a left/right spatial asymmetry, so a permutation-invariant model is structurally unable to solve it, whether the signal is motor or ocular.

### B2.3 Electrode displacement (test time only)
- **Simulated misplacement:** whole-cap rotations of 0–15° (≈ 0–24 mm at the vertex, for a head radius of ~9 cm), applied separately as a **lateral** shift (rotation about the nose–inion axis) and an **anterior–posterior** shift (rotation about the ear–ear axis). Signals at the displaced positions are interpolated from the true positions with spherical splines.
- **Prediction:** lateral shifts hurt more than AP shifts, because they break left/right symmetry: a lateral shift makes the electrodes on one side sample the EOG and the motor cortex differently from the other, which looks like a class difference and pushes predictions towards one class. For 5° (~8 mm, a typical placement error) the drop is small (≤ 3 points); at 15° lateral, 5–10 points. AP shifts cost ≤ 3 points even at 15°.
- **By model:** EEGNet and TF-time degrade similarly (both have one learned 64-weight spatial filter per map). TF-chan-id degrades at least as much, since its identity embeddings are tied to electrode indices whose content has moved. TF-chan-noid stays flat at chance.
- **Responsible weights:** the first spatial weights (EEGNet `conv_spatial`, TF-time `conv_spatial`, TF-chan-id `id_embed`). The smoother the learned spatial pattern, the smaller the drop; a model relying on a broad ocular field should be more robust than one relying on a focal central (Laplacian-like) pattern.

---

## B2.4–B2.5 (written after the B2.2 main runs finished; before any attribution, deletion, correlation or retraining)
Known at this point: mean test accuracy over 37 subjects, all training subjects, 3 seeds: EEGNet 79.4%, TF-time 77.5%, TF-chan-id 73.9%, TF-chan-noid 48.2%. So **TF-time is the best transformer** and is the B2.4 model. Scaling and displacement results were not yet available.

### B2.4 What TF-time relies on
- **Time:** grad × input importance peaks in the tokens covering **0.3–0.9 s** after the cue (the saccade toward the target; the F7−F8 class difference peaks at 0.4 s in A3) and falls off over later tokens. Attention is flatter over time than grad × input.
- **Space (input-level grad × input):** concentrated at lateral frontal/temporal sites (F7, F8, FT7, FT8, AF7, AF8, T7, T8). Its rank correlation with the A3 ocular (< 4 Hz) map is **positive (ρ > 0.5)**, and its correlation with the A3 mu/beta map is near zero.
- **Faithfulness:** deleting the top grad × input tokens hurts more than deleting random tokens; deleting the top 2 of 13 (15%) costs ≥ 10 points vs ≤ 3 points for 2 random tokens. Attention ranking is less faithful than grad × input, closer to random, because last-layer CLS attention reflects routing, not the size of each token's effect on the logit.
- **Mapping to Part A:** the model relies mainly on the A3 ocular route, not on sensorimotor ERD.

### B2.5 Link to Part A, and retraining without the ocular route
- **Correlations across the 37 test subjects:** network accuracy correlates with the A3 ocular decoder (ρ ≈ 0.5–0.7) more than with the A2 μ lateralisation index (ρ ≈ 0.2–0.3). With all predictors together, the ocular decoder is the one with a clear unique contribution.
- **Removal (drop 21 frontal/ocular sites, CAR over the rest, 8–30 Hz):** within-subject checks on the network inputs: slow time-locked amplitude ≈ 72% → ≈ 50%; posterior evoked ≈ 60% → 52–55% (some evoked power lies in 8–20 Hz); sensorimotor mu/beta power unchanged at ≈ 55%.
- **Retrained networks:** accuracy falls from ~79% to **≈ 58–62% (EEGNet)** and **≈ 55–58% (TF-time)**. What remains correlates more with the A2 μ lateralisation index (ρ ≈ 0.4–0.5) than the full-input models did, i.e. what is left is mostly motor.
