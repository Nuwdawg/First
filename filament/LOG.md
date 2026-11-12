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

**21:13** Simulated filtered ground truths on validation (`analysis/gt_subset.py`): keeping
only Left/Right filaments or only large ones moves the best confidence floor to 0.93,
which the floor-0.90 probe contradicts. A stricter match threshold of about 0.59 to 0.60
reproduces the three probe scores exactly (`analysis/tau.py`) and predicted 0.42 for the
consensus candidates (`analysis/tau_cands.py`).

**21:16 to 00:29** No work: usage limit.

## 11 November 2026

**00:29** Submitted G (58219268, three models, max x fraction >= 0.92) and F (58219271,
three models, mean >= 0.88).

**00:36** G scored **0.40**, F scored **0.41**; the team stayed 5th among the teams at
0.41. The eight-view consensus does not transfer: a validation gain of +0.007
(cross-validated) produced nothing visible in public. The stricter-match-threshold idea
is rejected with it (it predicted 0.42 for both).

**00:42** Consensus of earlier submissions (`analysis/blend41.py`, `analysis/blend.py`).
Fifteen files that each displayed 0.41 were downloaded; files agreeing above 0.97 with a
newer one were dropped, leaving eight distinct ones (F, transport-mean 58186186,
seven-view 58160222, ten-source 58159470, multimodel-exact 58127006 for the exp169
family, m2f-only 57816701, exp162b 56648050, deployed 56309562). Masks are clustered
across files at IoU 0.5; an object is kept if enough files contain it, with a
pixel-majority mask.

**00:50** `blend41_thr50` (58219681, at least 4 of 8 files, 1,156 masks) and
`blend41_thr60` (58219683, at least 5 of 8, 1,110 masks) both scored **0.41**, and the
team moved from 5th to **3rd** of the twenty teams at 0.41, passing two teams. One of
the two files is therefore a new best on the unrounded score. This is the first change
in position since before 9 November.

**00:59** `blend11_thr50` (58219979, the eight files plus candidates A, D, E; at least 6
of 11; 1,120 masks) scored **0.41** with no further change in position. No submissions
left on 11 November.

**01:03** File-level vote emulated on validation with fourteen rebuilt pipelines
(`analysis/blend_val.py`). Eight mixed files: 0.4778 (at least 2 agree), 0.4810 (4),
0.4784 (6), 0.4734 (7), against a best member of 0.4817. Six single-view files: 0.4754
at 4 of 6 against a best member of 0.4711. All fourteen: 0.479 to 0.482 from 3 to 8
agreeing. The vote sits on the same plateau as the eight-view consensus, and the number
of files required barely matters between a quarter and 60 %.

**01:13** Files prepared for 12 November (`analysis/blendw.py`): `blendW_thr50` and
`blendW_thr40` (22 distinct files weighted 1.0 / 0.6 / 0.3 by the public score they
displayed: 0.41 / 0.40 / 0.39; weighted share at least 50 % or 40 %; 1,138 and 1,179
masks) and `blend8_thr38` (the eight 0.41 files, at least 3 of 8; 1,197 masks). The
weighted code path reproduces `blend41_thr50` exactly. All these files agree with
`blend41_thr50` at 0.974 to 0.983, so they are small steps around the current best.

Six ideas have now been checked against leaderboard evidence and rejected: pooled
annotators (probe 0.39), sparser test annotations (probe 0.40), ground truth filtered by
class and by size (both contradict the floor-0.90 probe), a stricter match threshold and
transfer of the eight-view consensus (G 0.40, F 0.41).

**01:24 to 01:48** (not logged at the time) Vote variants built for the next day
(`analysis/blend_variants.py`). A confidence tiebreaker inside the file vote does not help
on validation (-0.002 to -0.0035 for eight mixed files, +/-0.003 for six single-view
files; `analysis/vote_conf.py`). Carrying the three-model consensus masks instead of the
file-majority masks gives +0.0015 [-0.0016, +0.0046] and +0.0023 [-0.0002, +0.0053]
(`analysis/vote_masks.py`): inconclusive.

**01:48 on 11 Nov to 14:29 on 12 Nov** No work: the session was stopped by the account's
usage limit and then by worker restarts. All of 11 November's remaining hours were lost.

## 12 November 2026

**14:29** `blendW_thr50` (58275876) scored **0.41**, no change in position. The team is
now 4th of 25 at 0.41 because a new team (Guillaume) entered just above it that morning.
Weighting the vote by each file's public score does not beat the plain vote of the eight
0.41 files.

**14:40** Seventh hypothesis rejected (`analysis/calib.py`): the consensus was tuned on
hold-out models but run on test with all-data models, so its floor might be miscalibrated
there. It is not: objects per photograph above each floor are 4 to 7 % lower on test than
on validation for every rule, single view and consensus alike.

**14:45** Geometric mismatch simulated on validation (`analysis/geom.py`): a rigid shift
of one pixel costs 0.006 to 0.028 and two pixels 0.05 to 0.09; a one-pixel dilation 0.030
and erosion 0.032. A mismatch of that kind in the hidden masks would look like the gap
everybody reports, but it is not supported: the organisers rasterise with pycocotools as
we do, and in September a one-pixel erosion of YOLO masks gained the same +0.03 in public
as on validation. Not probed.

**14:52** **The test set is triple-annotated** (`analysis/composition.py`). Copying one
annotator's record and scoring it with the official rule gives PQ 0.6798 on training
photographs with two annotator records and **0.5556** on photographs with three; 108
leaderboard teams (all using the public annotations) sit at exactly 0.55. Per-record
scoring with three records per photograph explains that cluster; a random mix of one,
two and three records would put a copied record near 0.8. See `docs/06_test_population.md`.

**14:55** Consequences measured on validation (`analysis/bysub3.py`, `bygroup.py`,
`groupstyle.py`). Triple-annotated photographs occur only in annotator groups 0101, 0102,
0104 and 0501 ("target" below; 623 of the 1,154 records). The deployed model scores 0.454
on the 27 triple-annotated validation photographs against 0.4705 overall. From the labels
alone, triple-annotated records have fewer filaments (6.64 against 7.80 per record),
larger ones (median 1,361 px against 1,087) and wider ones (mean width 10.3 px against
9.3). For target records a one-pixel erosion of our masks costs 0.049 and a dilation
0.020; for the other groups it is 0.016 and 0.041.

**15:00** Sub-pixel thickening (blur sigma 1 px, threshold 0.40, +3.5 % area;
`analysis/subthick.py`, `blend_thick_val.py`): on target validation records +0.0026
(three-model consensus), +0.0022 (file vote), -0.001 (single view); on triple-annotated
photographs +0.0047, +0.0034, 0.000. `blend41_thr50_thick40` (58276674,
`analysis/thicken_sub.py`) scored **0.41** with no change in position.

**15:07** Where the target annotators differ most (`analysis/splitmerge.py`): our mask is
only a piece of the annotated filament for 9.1 % of their masks (4.8 % for the others),
and 12.3 % of our predictions are such pieces and count as false positives (6.3 %).
Swapping a piece for the model's own larger low-confidence candidate makes things worse
on every population (`analysis/larger.py`); masks at a lower probability threshold change
nothing (`analysis/lowmask.py`); none of the eight cached models suits the target
annotators better than the deployed one (`analysis/model_by_pop.py`, 0.44 to 0.47).
The style has to be learned in training.

**15:24** Kaggle kernel `cosmicwitness/sol-exp202-target-ft-t4` pushed
(`exp202_target_ft/`): the deployed weights (exp139 epoch 40) fine-tuned on the 527
training records of the four target groups, two runs in parallel (learning rate 2e-5 and
5e-5, six epochs, cosine, evaluation every second epoch), checkpoints ranked on the 96
target validation records, eight-view candidates cached for the best checkpoint and
single-view test candidates for the rest. The group filter, the checkpoint-aware cache
builder and the selector were smoke-tested locally on CPU; the selector reproduces
0.47052 / 0.45702 / 0.45392 (all / target / triple) for the deployed model.

**15:27** `blend41_thr60_thick40` (58277308) scored **0.41**, no change in position. With
the 15:00 result this bounds the public value of the thickening below the gap to the next
team on whichever of the two vote files is our best. Dropped.

**15:33** `cand_C_e40_d4_maxfrac_092` (58277481), the eight-view consensus of the
deployed model alone, scored **0.40**: below the single view (0.41) although it is better
on validation by 0.008 (all), 0.014 (target records) and 0.016 (triple-annotated). The
test-side eight-view caches are sound (`analysis/viewcheck.py`: each rotated or flipped
view finds 99.9 % of the confident single-view masks at median IoU 0.914 on test, 0.913
on validation, with the same sub-pixel offsets). The public half is 90 photographs;
a difference of half a point between two rules is within its sampling noise.

**15:40** Minimum area is not a lever on any population (`analysis/minarea_pop.py`).

**15:42** Label-fusion ceiling on the 151 triple-annotated training photographs, labels
only (`analysis/fuse_oracle.py`). Copying one annotator: PQ 0.554 to 0.556. Keeping the
filaments at least two of three drew, with a fused mask: 0.585 to 0.587 (majority pixels
or the smaller medoid; union 0.559, intersection 0.538). Adding the masks only one
annotator drew: 0.47 to 0.50. Only the filaments all three drew: 0.428. Of 3,006 masks,
35.5 % were drawn by one annotator only, 28.1 % by two and 36.3 % by all three.

**15:42 to 19:28** No work: usage limit. The kernel kept running.

**19:30** exp202 results (`docs/07_exp202_results.txt`, kernel time 2.5 h). Fine-tuning on
the target groups' records does not help. Single view, best floor, target validation
records: 0.4511 / 0.4533 / 0.4538 (learning rate 2e-5, epochs 2 / 4 / 6) and 0.4525 /
0.4545 / 0.4513 (5e-5), against 0.4570 for the deployed model; on all validation records
0.460 to 0.463 against 0.4705. Eight views of the selected checkpoint (5e-5, epoch 4):
0.4650 to 0.4657 on target records against 0.4675 to 0.4735 for the deployed model;
three-model consensus with it: 0.4780 against 0.4771. The disagreement with the target
annotators is disagreement among the three annotators of a photograph, not a style the
model failed to learn.

**19:33** Changing only the mask source inside the file vote changes nothing on target
records (`analysis/maskswap.py`: three-model masks -0.0006, eight-view -0.002,
single-view -0.005). The deployed selection with three-model masks is +0.0047 on target
records and +0.006 overall.

**19:40** `blend8_thr38` (58283396, at least 3 of 8 files, 1,197 masks) scored **0.41**,
no change in position. Five submissions used on 12 November: 0.41, 0.41, 0.41, 0.40,
0.41. Position: 4th of 25 at 0.41 (overall 138); one team at 0.42.

**Standing on the evening of 12 November.** Tested and without visible public gain:
consensus across views and models (two of three variants below the single view),
weighting the vote by public score, more or fewer agreeing files, thicker masks, mask
source, minimum area, larger candidates, lower mask thresholds, other existing models,
fine-tuning on the test-side annotators. A leaderboard comparator built from deliberately
wrong masks would resolve changes of a few ten-thousandths; it has not been used because
it is leaderboard probing, and the user has been asked whether they want it.

**20:15 to 20:50** Candidate pool for the remaining fifteen submissions
(`docs/08_candidate_pool.txt`, `analysis/tickets13.py`, `pool13.py`, `members13.py`).
Files that agree with the best one at about 0.90 have scored on both sides of 0.405
(G and C at 0.40; F and the single view at 0.41), so on 90 public photographs two sound
files that differ in 150 masks scatter by about 0.004. Near-copies of the best file
cannot reach 0.415; a comparably good file built differently can. About 25 rules were
replayed on validation with the code that writes the test file: votes of single-view
models (nine models including the two fine-tuned ones), eight-view consensus per model,
votes of eight-view files, mean-score consensus across two to five models, and the
deployed selection with consensus masks. All score 0.459 to 0.471 on triple-annotated
validation photographs (deployed 0.454, vote replay 0.468). Agreement with the best test
file ranges from 0.87 to 0.95.

Planned for 13 November, one at a time with a position check: `voteS9_k4.csv` (nine
single-view models, at least 4), `pool13/five85.csv` (five-model mean-score consensus,
floor 0.85), `deployed_sel_three_masks.csv`, `pool13/d4_swinb.csv`,
`pool13/S6noe40_k3.csv`; the last two may change with the first results.

**20:54** Kernel `cosmicwitness/sol-exp203-fused-ft-t4` pushed (`exp203_fused_ft/`):
the deployed weights fine-tuned on fused consensus labels. For each of the 238
multi-annotated training photographs with any agreement, one target record made of the
filaments at least two annotators drew (1,126 masks; 1,629 masks drawn by one annotator
only are dropped), each represented by a real annotator mask. Run C: the fused records
only, 12 epochs; run D: the 352 single-annotated records plus the fused records three
times, 3 epochs; learning rate 5e-5. Smoke-tested locally.

**20:55** Ranking headroom on the 27 triple-annotated validation photographs
(`analysis/rank_headroom.py`). Of the deployed model's objects above the floor, 73 % are
matched by at least two annotators, 12.7 % by one and 14.5 % by none; the confidence
separates "at least two" from the rest with AUC 0.93. A perfect selection among the
model's own masks would score 0.501 against 0.454 (three-model consensus: 0.518 against
0.472). A label-free comparison of what the eight-view consensus drops and adds shows the
same pattern on validation and test (`analysis/dropcheck.py`).

**21:00** **Correction: the test set is not all triple-annotated.** The bound recorded on
10 November (08:45) had dropped out of view: the published totals leave at most 368
annotator records for the 180 test photographs, 2.04 per photograph, where triples would
need 540; and 0.5556 would be displayed as 0.56, not 0.55. `docs/06_test_population.md`
now opens with the withdrawal. Consequences: the whole validation set is the yardstick
again (the candidate pool was already scored on it: `docs/08_candidate_pool.txt`, first
column); the fine-tune on four annotator groups was aimed at a population the test set is
not restricted to (it gave no gain on that population either); the thickening had no
basis on the test set, which agrees with its null result in public.
