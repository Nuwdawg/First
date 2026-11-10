# Candidate submissions for 11 November

Every file below was written by `analysis/make_d4_sub.py` or
`analysis/make_multi_sub.py`. Each script first replays its rule on validation through
the same function that writes the test file (`analysis/decode.py`) and prints the score,
so the validated rule and the submitted rule cannot drift apart. All files pass the
format check: 180 photographs, no empty masks, no overlapping masks.

Validation is per-annotator PQ on the 106 hold-out photographs. "Twin" means the test
file uses a model trained on all 707 photographs, whose validation score cannot be
measured; the number quoted is for the same recipe trained on 601 photographs.

| id | rule | validation PQ | test masks |
|---|---|---|---|
| deployed | e40, single view, floor 0.85 | 0.4705 | 1,192 |
| A | e40, eight views, mean of 8 scores >= 0.85 | 0.4771 | 1,130 |
| B | e40, eight views, max x fraction >= 0.90 | 0.4797 | 1,215 |
| C | e40, eight views, max x fraction >= 0.92 | 0.4787 | 1,102 |
| D | all-data Swin-S, eight views, mean >= 0.88 | 0.4766 (twin) | 1,093 |
| E | all-data Swin-B, eight views, mean >= 0.88 | 0.4758 (twin) | see log |
| F | e40 + all-data Swin-S + all-data Swin-B, eight views each, mean of the three model means >= 0.88 | 0.4817 (twins) | 1,051 |
| G | same three models, mean of the three max x fraction scores >= 0.92 | 0.4824 (twins) | 1,114 |

For F and G the paired bootstrap interval against the deployed model is
[+0.003, +0.021] on validation (twins), the first interval in this project's consensus
work that excludes zero. Pairs of the three models score 0.4776 to 0.4802.

The all-data models stand in for their twins without changing the rule's behaviour: on
validation the twins match 94 to 96 % of e40's confident objects, on test the all-data
models match 96 to 98 %, and F emits 6.19 masks per validation photograph against 5.84
per test photograph, the same ratio as the deployed model shows between the two sets
(`analysis/twin_check.py`).

## How these differ from the earlier ensembles

The earlier multi-source submissions (validation 0.489 to 0.495, public 0.40 to 0.41)
mixed YOLO models and weight blends with the Mask2Former models, and their member sets,
weights and thresholds were chosen from very large searches on the same validation
photographs. F and G use three models of one family, one aggregation rule fixed before
looking at combinations, and one tuned number (the floor), which sits on a flat part of
its curve (0.4816 at 0.85, 0.4817 at 0.88 for F).

## Expectation

The deployed model's unrounded public score is between 0.405 and 0.412. Three of its
equal-validation variants (eight-view fusion exp144, weight soup, calibrated ensemble
with YOLO) displayed 0.40, which suggests it sits in the lower part of that range. If
the validation gain of F or G (+0.011) transferred fully, they would land at 0.416 to
0.423 and display 0.42. The earlier vote-mask rules (exp162, exp169) transferred at most
about two thirds of their gain. A realistic expectation is 0.412 to 0.420: better than
even odds of staying at 0.41 on any single try, with a real chance of 0.42.

Order planned: G, then F, then C, with the last two slots decided by what the first
three show (including the team's position among the teams displaying 0.41, which moves
only when a submission beats the current best).
