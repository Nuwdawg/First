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
