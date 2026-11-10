# The pooled-annotator scoring hypothesis

## Observation

Of the 707 training photographs, 296 were labelled independently by two or three
annotators. The organisers' self-evaluation notebook scores a photograph's predictions
separately against each annotator's record ("per-annotator" scoring below). Everybody,
including this project until now, validates that way.

Annotators disagree strongly. Scoring one annotator's masks against another's on the same
photograph gives PQ 0.340 (SQ 0.634, RQ 0.536) over all ordered pairs
(`analysis/interann.py`).

## Hypothesis

On the leaderboard the ground truth of a photograph is the *pool* of every annotator's
masks under one image id, matched one-to-one with the predictions. This would happen
naturally if the solution file used ids of the form `<photo>_<n>` (the same form the
submission uses) and the scorer grouped rows by the part before the underscore. The
organisers confirmed that the leaderboard uses one-to-one matching where the notebook
does not, and declined to describe the multi-annotator handling.

Because predicted masks may not overlap, a prediction can match only one of the two or
three masks drawn for the same filament; the others are counted as misses. With P
predictions and N pooled ground-truth masks the score reduces to

    PQ_pooled = 2 * sum over predictions of m(pred) / (P + N),

where m(pred) is the best IoU with any annotator's mask if it exceeds 0.5 and zero
otherwise. Each prediction costs one half in the denominator and earns its best IoU with
*any* annotator; under per-annotator scoring it is instead judged against each annotator
in turn.

## Evidence so far

1. **The size of the gap.** The deployed model scores 0.4705 per-annotator on validation
   and 0.4002 pooled (`analysis/valscore.py`). Its public score is 0.41. The pooled
   figure reproduces the 0.06 to 0.08 gap that every submission and every team has seen.
2. **No per-annotator mixture explains it** (see `01_prior_state.md`).
3. **Earlier validation gains.** The ensemble rules found before raised per-annotator PQ
   mainly by removing predictions that one annotator did not draw (for example "11 fewer
   true positives, 25 fewer false positives"). Under pooled scoring many of those
   removed predictions match another annotator, so the gain disappears.

## The two protocols disagree about the confidence floor

Deployed model, single view, validation (`analysis/sweep3.py`):

| floor | predictions | per-annotator PQ | pooled PQ |
|---|---|---|---|
| 0.90 | 627 | 0.4625 | 0.3781 |
| 0.85 (deployed) | 756 | **0.4705** | 0.4002 |
| 0.80 | 833 | 0.4654 | 0.4054 |
| 0.75 | 883 | 0.4598 | 0.4076 |
| 0.70 | 950 | 0.4537 | **0.4091** |
| 0.65 | 1000 | 0.4464 | 0.4063 |
| 0.60 | 1045 | 0.4390 | 0.4027 |
| 0.50 | 1121 | 0.4273 | 0.3960 |
| 0.30 | 1295 | 0.3953 | 0.3771 |

Lowering the floor from 0.85 to 0.70 costs 0.017 under per-annotator scoring and gains
0.009 under pooled scoring.

## Probe

Submission 58193731 (10 Nov, 08:35 UTC) is the deployed model with the floor at 0.70 and
nothing else changed (1,526 masks instead of 1,192). The rebuilt floor-0.85 file matches
the original 0.41 submission mask for mask on all 180 photographs, so the floor is the
only difference.

| outcome | reading |
|---|---|
| 0.41 or 0.42 | pooled scoring; validate everything with the pooled metric from now on |
| 0.39 or 0.40 | per-annotator scoring; the gap has another cause |

Result: see `LOG.md`.
