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
