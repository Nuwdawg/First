# State of the project on 10 November 2026

**Leaderboard.** Team "supersede" has a best public score of 0.41 (50 submissions), fifth
of nineteen teams displaying 0.41; one team displays 0.42. Scores of 0.49 and above come
from the publicly available test annotations (106 teams sit at exactly 0.55) and are not
a target.

**Deployed model.** Mask2Former with a Swin-S backbone, trained and run at the native
2048 x 2048 resolution (experiment exp139, epoch 40). One forward pass, candidates scored
by class probability times mean mask probability, painted in descending score with a floor
of 0.85 and a minimum area of 100 px. Validation PQ 0.4705 (TP 881, FP 361, FN 444) on
106 photographs held out by photograph; public score 0.41.

**What had been tried.** U-Nets, four YOLO-seg systems, Mask2Former variants, test-time
augmentation, mask refiners, verifiers, weight soups, calibrated ensembles, and then (in
the sessions recorded in `for shame.txt`) several hundred million ensemble-selection
configurations over up to 78 prediction sources, split rules, learned selectors,
semantic-evidence scorers and weight transport. The best validation PQ reached 0.4949.
Every one of these scored 0.40 or 0.41 in public.

**The recurring symptom.** Every model loses 0.06 to 0.08 between validation and the
public leaderboard, and other teams report the same gap in the forum (0.487 locally,
0.41 in public). Validation gains of 0.01 to 0.02 never appeared on the leaderboard. The
experiment book attributed the gap to "ordinary generalisation error".

**Why that explanation does not hold.** The test photographs are a random draw from the
same pool as the training photographs: same years, same six stations, and the same
distance in time to the nearest training photograph (median 1.5 days for both). The
deployed model's validation PQ, broken down by how many annotators labelled a photograph,
is 0.469 (one), 0.499 (two) and 0.454 (three), and by annotator group it ranges from 0.43
to 0.54 for every group with more than ten records. No mixture of these reaches 0.41. The
gap therefore comes from how the test set is scored, which is the subject of
`02_scoring_protocol.md`.
