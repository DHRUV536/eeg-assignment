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
