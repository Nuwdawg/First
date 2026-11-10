# Experiment log

Times are UTC. "Per-annotator" and "pooled" PQ are defined in `docs/02_scoring_protocol.md`.
Validation is the project's 106-photograph hold-out (180 annotator records, 1,325 masks).

## 10 November 2026

**08:11** Goal set: public score 0.42. Read `for shame.txt`, the competition pages, all 51
forum threads, the organisers' scoring notebook, and the project's experiment book and
report (Kaggle datasets `sol-book-src`, `sol-paper-src`).

**08:20** Data audit (`analysis/explore1.py`). 707 training photographs, 1,154 annotator
records, 8,199 masks, 37 annotator batches; 180 test photographs from the same years and
stations, with the same time gap to the nearest training photograph (median 1.5 days).

**08:22** Annotator agreement (`analysis/interann.py`): PQ 0.340 between annotators of the
same photograph, from 0.25 (batches 010401/010402) to 0.44 (020401/020403).

**08:26** Reproduced the deployed model's validation score exactly: 0.470525, TP 881,
FP 361, FN 444 (`analysis/valscore.py`). The same predictions score 0.4002 when the
annotators of a photograph are pooled and matched one-to-one.

**08:31** Confidence-floor sweep under both protocols (`analysis/sweep3.py`); table in
`docs/02_scoring_protocol.md`. Per-annotator optimum 0.85, pooled optimum 0.70.

**08:33** Rebuilt the deployed test submission from the saved candidate cache; identical
to submission 56309562 on all 180 photographs (1,192 masks).

**08:35** Submitted probe 58193731: deployed model, floor 0.70 (1,526 masks). One
submission left today.

**08:40** Probe 58193731 scored **0.39**. The pooled-annotator hypothesis is rejected:
the leaderboard moved as per-annotator scoring predicts. The offset between validation
and public score for the deployed model family is 0.059 to 0.066.

**08:42** Other protocol variants (`analysis/protocols.py`): mean of per-record PQ 0.4826,
mean of per-photograph PQ 0.4784, first record only 0.4674. None explains the offset.

**08:43** Annotator batches are assigned independently of capture time
(`analysis/batchtime.py`): the nearest photograph in time shares a batch 4 to 10 % of the
time against 6 % by chance. A test photograph's annotators cannot be anticipated.

**08:45** Published dataset totals (Ahmadzadeh et al., Scientific Data 2024: 10,244
filaments in 1,593 annotator records of 958 photographs) against the training file
(8,199 filaments in 1,154 records of 707 photographs) leave at most 2,045 filaments in
439 records for the 251 photographs outside the training file: 4.7 filaments per record
against 7.1 in training. The model detects nearly as many filaments per test photograph
(6.6 at floor 0.85) as per validation photograph (7.1). If the totals are comparable, the
test annotators drew about two thirds as many filaments as training annotators did on
similar images. This uses only aggregate counts printed in the paper's abstract, no
annotations.

**08:46** Validation split by how many filaments the annotator drew relative to the
model's detections (`analysis/density2.py`). Records with a ratio below 0.7 (27 records)
score 0.417 at floor 0.85 and 0.481 at 0.93; records with a ratio of 1.25 or more score
best at 0.70. A sparser test set would favour a stricter floor.

**08:48** Submitted probe 2 (file `probe_e40_floor090.csv`): deployed model, floor 0.90
(997 masks). Validation predicts 0.40; a sparser test set predicts 0.42 or more. No
submissions left today.

**08:49 to 13:30** No work: the session's worker process was restarting. Four hourly
check-ins went unanswered.

**13:31** Probe 2 (58194117, floor 0.90) scored **0.40**, the value plain validation
predicts (0.4625 minus the offset). The sparser-test-set idea gives no gain from the
floor and is set aside.

What the two probes establish for the deployed single-view model:

| floor | validation PQ | public | implied offset |
|---|---|---|---|
| 0.70 | 0.4537 | 0.39 | 0.059 to 0.069 |
| 0.85 | 0.4705 | 0.41 | 0.056 to 0.066 |
| 0.90 | 0.4625 | 0.40 | 0.058 to 0.068 |

All three are consistent with one offset between 0.059 and 0.066, so the deployed
model's unrounded public score is about 0.4085 and a displayed 0.42 needs about +0.007.
The earlier ensembles (validation 0.484 to 0.495, public 0.41) had offsets of at least
0.074: no more than half of their validation gain was real.

**13:35** Progress email 1 sent.

**13:38** Eight Mask2Former-family models with saved validation and test predictions
loaded (`analysis/zoo.py`). Alone: e40 0.4705, ctrl7 0.4674, ctrl31 0.4673, head 0.4626,
focal 0.4607, paste 0.4548 (base20 and qual31 use a different score scale).

**13:40** Six-model consensus: 0.4701. No gain over the single deployed model.

**13:42** Near-miss analysis: 199 annotated masks are touched but unmatched; the model's
shape is the smaller one in 42 % and the larger one in 46 %.

**13:43** Eight-view consensus of e40: 0.4771 (mean of 8) and 0.4797 (max x fraction).
A 2560-pixel view and a second model add nothing.

**13:46** Label-free comparison of validation and test photographs: no difference.
The consensus gain is concentrated on multi-annotator photographs.

**13:49** Leave-one-annotator-out ceiling: selecting candidates with the other
annotators' masks scores 0.447 on the held-out annotator; the deployed rule scores
0.4712 on the same records.

**13:50** Consensus gain replicated on base20 (+0.019), qual31 (+0.005) and the Swin-S
hold-out model (+0.009); shifted views add nothing. Summary in
`docs/03_ceiling_and_consensus.md`.

**13:54** Candidates A, B, C built from e40's eight views; validation replays 0.4771,
0.4797, 0.4787. Cross-fitted learned rankers of the eight-view clusters score 0.4732 to
0.4813 (`analysis/learned_rank.py`), no clearer than the hand rules, so the rules stay.

**13:57** Swin-B hold-out model: single view 0.4711, eight-view consensus 0.4758 / 0.4767.
Sixth model on which the consensus gain appears.

**13:58** Three-model consensus (e40 with the Swin-S and Swin-B hold-out twins, eight
views each): 0.4817 (mean) and 0.4824 (max x fraction), +0.011 to +0.012 over the
deployed model, bootstrap interval [+0.003, +0.021] (`analysis/three.py`).

**14:01** Painting policy for partly covered candidates: no effect (+0.0002 to +0.002).

**14:04** Per-instance mask threshold chosen by approximate expected IoU (Nowozin 2014),
using the cached masks at logits -1, 0 and +1: always selects the logit-0 mask; the
oracle best-of-three gains 0.008 IoU. The masks are too sharp for the threshold to
matter (`analysis/eiou.py`).

**14:06** Candidates D, E, F, G built for the test set with all-data models in place of
the twins; label-free twin check passed. See `docs/04_candidates.md`.

**14:07 to 19:28** No work. Cause identified at 19:30 from the session's rate-limit
record: the account's five-hour usage limit, reached after about 37 minutes of work in
each window (the 08:49 pause had the same cause). Six hourly check-ins went unanswered.

**19:35** Progress email 2 sent, explaining the limit and the realistic email cadence.
`STATE.md` added so work can resume from the repo alone.

**19:31** Single-stage clustering of all 24 views (three models x eight): 0.4818 (mean),
0.4792 (max x fraction), 0.4823 (mean of per-model max x fraction). Same plateau as the
two-stage rule (`analysis/onestage.py`).

**19:33** Native 1024-px tile views of e40: 0.446 as a score on their own; blending them
into the eight-view score changes nothing (0.4772 / 0.4785). No GPU job for test tiles
(`analysis/tiles.py`).

**19:35** Value of knowing the annotator (`analysis/style_oracle.py`), in-sample upper
bounds from choosing the confidence floor with the labels: per annotator batch 0.4877
(single view) and 0.5033 (three-model consensus); per photograph 0.5037 / 0.5168; per
record 0.5191 / 0.5346. Even perfect knowledge of each annotator's inclusion habit is
worth about +0.02, and it cannot be known for a test photograph. An annotator-conditioned
model was therefore not built.

**19:35** Nested 5-fold cross-validation of the three-model rule family (aggregator and
floor chosen on four fifths of the photographs, scored on the rest, 200 repeats):
+0.0068 over the deployed model (range +0.002 to +0.010 across repeats). The in-sample
figure of +0.011 to +0.012 is therefore optimistic by about 0.004.

**19:38** Variants built: G94 (three-model, max x fraction, floor 0.94; validation 0.4713,
941 masks) and F85 (mean, floor 0.85; 0.4816, 1,173 masks). Agreement between candidate
test files, as PQ of one scored against another (`analysis/diversity.py`): F vs G 0.954,
F vs deployed 0.840, F vs the earlier seven-source submission 0.878, single-model
candidates A, D, E against each other 0.865 to 0.898.

**19:40** Checked whether the training file splits multi-piece filaments into separate
instances (which would make the hidden test masks coarser than the training ones): it
does not. All 8,199 annotation ids are unique, spines are unique within a record (3
exceptions), and 99.8 % of spine points lie inside their own mask.

**19:44** Mask hygiene (`analysis/figs.py`, `analysis/components.py`). Annotated masks are
single connected pieces (99.8 %); 14 % of deployed masks and 12 % of candidate G's have
two or more pieces. The extra pieces are specks: median 2.3 % of the mask's area, median
gap 8.6 px. Keeping only the largest piece (-0.0006 single view, -0.003 consensus),
splitting pieces into separate filaments (-0.001 to +0.003 depending on thresholds) and
bridging the gap (-0.002 to -0.006) do not help, so masks are left as they are.
Figure `docs/figures/test_candidate_G_vs_deployed.png` shows candidate G against the
deployed predictions on six test photographs; the masks follow the dark filament bodies
and the two agree closely on the main filaments.

**19:46** Sensitivity checks on the three-model consensus (validation 0.4824).
One-pixel erosion costs 0.038 and one-pixel dilation 0.030 (`analysis/thick.py`), so
mask thickness is tuned to the training annotators to well under a pixel. A rigid shift
of all masks costs 0.008 (up), 0.027 (down), 0.013 (right), 0.022 (left), 0.07 to 0.09
for two pixels (`analysis/shift_sens.py`).

**19:48** The up/down asymmetry is the same in all three single-view models (-0.006 up,
-0.028 down), so predictions sit about a third of a pixel below the rasterised
annotations. A fractional correction applied inside the vote raises SQ from 0.6894 to
0.6909 but PQ by only +0.0004; the opposite shift costs 0.0025
(`analysis/subpixel.py`). Real but too small to matter; not adopted.

**19:56** Where the consensus gain comes from (`docs/05_gain_breakdown.txt`): positive at
five of six stations (+0.010 to +0.028; Udaipur -0.006 to 0.000), positive in 2011-2016
and negative in 2017-2022 (36 records, -0.011 to -0.025), larger on photographs with two
or three annotators (+0.010 to +0.018) than with one (+0.005), and absent on photographs
with four or fewer filaments. On resampled validation sets candidates F and G beat the
deployed model 99 % of the time and each other about half the time.

**19:59** Nested cross-validation over a broader family of 288 configurations (model
subsets, link threshold, vote threshold, aggregator, floor; `docs/06_family_nested_cv.txt`):
whole family +0.0047 over the deployed model; three models with fixed link and vote
+0.0069; e40 alone +0.0035; four models (adding ctrl31) +0.0085. The honest validation
gain of the consensus is +0.005 to +0.008, and the settings do not matter within the
plateau.

**20:01** Candidate H built: four models (e40, all-data Swin-S, all-data Swin-B, ctrl31),
max x fraction, floor 0.90.
