# Who labels the test set (12 November 2026)

> **Correction, 12 November 21:00 UTC. The conclusion of this page is withdrawn.**
> The inference below (every test photograph has three annotator records) contradicts a
> bound already recorded on 10 November (`LOG.md`, 08:45): the published dataset has
> 10,244 filaments in 1,593 annotator records of 958 photographs, the training file has
> 8,199 filaments in 1,154 records of 707 photographs, so the 251 photographs outside the
> training file hold at most 439 records, and the 180 test photographs at most 368, that
> is **at most 2.04 records per photograph**. A test set of triples would need 540. The
> match with the 0.55 cluster is also weaker than stated: 0.5556 would be displayed as
> 0.56, where only six teams sit. The test set is therefore a mix of one, two and three
> records per photograph much like validation (1.70), and the whole validation set, not
> its triple-annotated part, is the relevant yardstick. What remains valid on this page:
> the description of the annotator groups in the training data, the label-fusion ceiling
> for multi-annotated photographs, and the negative results (thickening, larger
> candidates, fine-tuning on the four groups). What does not: any claim that the test
> set is scored by those four groups, and the reading of the validation gap as a
> composition effect.

## The observation

108 teams sit at exactly 0.55 on the public leaderboard. They use the publicly available
annotations, so their file is a copy of somebody's ground truth, yet it scores 0.55 and
not 1.0. Under the organisers' rule a photograph is scored against each annotator's
record separately. On the training data, copying one annotator's record and scoring it
that way gives (`analysis/composition.py`):

| photographs with | PQ of a copied record |
|---|---|
| one annotator record | 1.000 |
| two records | 0.6798 |
| three records | **0.5556** |

A test set drawn like the training set (58 % one record, 21 % two, 21 % three) would put
a copied record near 0.8. A test set in which every photograph has three records puts it
at 0.5556. This is indirect evidence, but it is the only composition that fits.

## What follows

1. **Part of the validation gap is composition.** The deployed model scores 0.4705 on the
   validation mix and 0.454 on its 27 triple-annotated photographs; three people rarely
   draw the same filament the same way (they agree with each other at PQ 0.333).
2. **The annotators are known.** In the training data triple-annotated photographs occur
   only in groups 0101, 0102, 0104 and 0501 (25, 13, 70 and 43 photographs). Photographs
   are assigned to groups without regard to date or station, so the test photographs
   cannot be assigned to a group, but the population is these twelve annotators.
3. **They have a style** (`analysis/groupstyle.py`, labels only; `analysis/bygroup.py`,
   `analysis/splitmerge.py`, model against labels):

| | target groups | other groups |
|---|---|---|
| filaments per record (triple-annotated against single) | 6.64 | 7.80 |
| median filament area, px | 1,361 | 1,087 |
| mean filament width, px | 10.3 | 9.3 |
| cost of eroding our masks by one pixel | 0.049 | 0.016 |
| cost of dilating our masks by one pixel | 0.020 | 0.041 |
| annotated masks missed because our mask is only a piece | 9.1 % | 4.8 % |
| our predictions that are a piece of an annotated mask | 12.3 % | 6.3 % |
| deployed model, validation PQ | 0.457 | 0.484 |

The model was trained on all 974 training records, 46 % of them from annotators who never
score a test photograph and who draw thinner, shorter and more numerous filaments.

## What was tried with it

- Sub-pixel thickening of the best file (+0.002 to +0.005 on the target validation
  records): 0.41 in public, no change in position.
- Swapping pieces for larger candidates, lower mask thresholds, a different existing
  model: no gain on validation.
- Fine-tuning the deployed weights on the target groups' records only (exp202, Kaggle
  GPU, 2.5 h): no gain. Best checkpoint 0.4545 on target validation records against
  0.4570 for the deployed model; 0.462 against 0.4705 overall
  (`docs/07_exp202_results.txt`).

## Why the fine-tune changed nothing: the ceiling for three annotators

With the three records of a photograph in hand, what single prediction scores best?
(`analysis/fuse_oracle.py`, 151 photographs, labels only.)

| prediction | PQ |
|---|---|
| copy one annotator | 0.554 to 0.556 |
| filaments at least two of three drew, fused mask | 0.585 to 0.587 |
| the same plus the masks only one annotator drew | 0.47 to 0.50 |
| only the filaments all three drew | 0.428 |

35.5 % of the masks on these photographs were drawn by one annotator only; each such mask
is right once and wrong twice. A model that sees only the image scores 0.41 to 0.45
against a labels-in-hand ceiling of about 0.59. The "pieces" in the table above are
mostly the three annotators of one photograph choosing different extents, which is why
training on their labels alone moved nothing, and why the honest part of the leaderboard
is packed between 0.39 and 0.42.

## Caveats

The composition is inferred, not confirmed. The triple-annotated validation sample is 27
photographs (81 records), so differences below about 0.01 on it are not reliable. If the
test photographs were labelled by other groups, the style numbers above do not apply.
