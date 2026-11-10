# Rules that constrain the work

Sources: the competition pages (Overview, Evaluation, Rules, Data) and the organisers'
replies in the discussion forum, read on 10 Nov 2026.

**Allowed**
- Any algorithm, including ensembles; AI assistance, provided the team understands and can
  reproduce every step.
- All metadata in the training file (spine, bbox, area, category) as training signal.
- External *unlabelled* data for training (other GONG frames, other instruments,
  magnetograms), provided no frame is within one hour of a test timestamp.
- Transferring training annotations to nearby unlabelled GONG frames, for training only.

**Forbidden**
- Any ground truth other than the provided training annotations. The public MAGFiLO
  release contains annotations for all 180 test photographs; using it in any form
  disqualifies the entry. SWEFil and other labelled filament sets are also excluded.
- Any input at inference other than the supplied test image (no neighbouring frames).
- Overlapping predicted masks: the scorer raises an error.
- Hand labelling of test images.

**Scoring**
- Panoptic Quality with a match threshold of IoU > 0.5, pooled over the test set.
- The public leaderboard uses about half of the 180 test photographs.
- Five submissions per day; two final selections.
- The organisers declined to say how photographs with several annotators are scored on
  the test set ("some subtle differences that we do not wish to reveal"). See
  `02_scoring_protocol.md`.

**Final judging** is 70 % quantitative and 30 % qualitative, and requires a public
repository, stored weights, a notebook that reproduces the predictions and a four-page
report. A leaderboard position alone does not decide the outcome.
