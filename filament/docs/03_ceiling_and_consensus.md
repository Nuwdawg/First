# Why ensembles stall, and the one change that holds up

All numbers are per-annotator PQ on the 106-photograph validation split. The deployed
model (Mask2Former Swin-S, epoch 40, single view, floor 0.85) scores 0.4705.

## The models are at a label-noise ceiling

**Annotators disagree more than the model disagrees with them.** One annotator's masks
scored against another's on the same photograph give PQ 0.340. The model scores 0.4705
against each of them.

**Knowing what other annotators drew does not beat the model's own confidence.** On the
47 validation photographs with several annotators, candidates were ranked by their
overlap with the *other* annotators' actual masks and scored on the held-out annotator
(`analysis/ceiling.py`). That selection reaches 0.447. The deployed rule, which sees no
annotation, reaches 0.4712 on the same records. Selection among the existing candidates
has no room left that an annotator-agnostic method could reach.

**Combining separately trained models adds nothing.** Eight Mask2Former-family models
have saved validation and test predictions. Alone they score 0.455 to 0.4705. A
consensus of six of them (mean confidence, majority-vote masks) scores 0.4701
(`analysis/zoo_cons.py`). Agreement across models as a stability signal scores 0.467 to
0.471 (`analysis/stability.py`). This is why the earlier ensemble searches found only
selection noise: the models make the same "errors" against any given annotator, and
those errors are the annotator's choices.

**Near misses are balanced.** Of 1,325 annotated masks, 881 are matched, 245 are not
touched by any prediction and 199 are touched without reaching IoU 0.5
(`analysis/nearmiss.py`). Among those 199, the prediction is the smaller shape in 42 %
and the larger in 46 %; matched pairs have a median area ratio of 1.03. The model sits
at the median granularity and thickness, so no global shift helps.

**Other things checked and discarded.** A 2560-pixel view scores 0.460 alone and does not
add to the consensus. Decoder-layer agreement and mask-sharpness scores do not beat the
deployed score (`analysis/stab_single.py`). Validation and test photographs are
indistinguishable by model confidence, view agreement, object size, brightness, contrast
and sharpness (`analysis/labelfree.py`), so the test images are not harder.

## Eight-view consensus of a single model

Each model is run on the eight flips and rotations of a photograph. Candidates from the
eight views are clustered by mask overlap (IoU at least 0.5); a cluster's mask is the
pixel-wise vote of its members (at least 60 %), and its score aggregates the eight view
scores, counting a view that did not find the object as zero.

| model | single view | consensus, mean of 8 | consensus, max x fraction of views |
|---|---|---|---|
| e40 (deployed) | 0.4705 | 0.4771 (+0.0065) | 0.4797 (+0.0092) |
| ctrl31 | 0.4673 | 0.4737 (+0.0064) | 0.4724 (+0.0051) |
| base20 | 0.4486 | 0.4677 (+0.0191) | 0.4645 (+0.0159) |
| qual31 | 0.4587 | 0.4641 (+0.0054) | 0.4599 (+0.0012) |
| Swin-S hold-out | 0.4677 | 0.4766 (+0.0089) | 0.4775 (+0.0098) |

The gain appears in all five models, which is the evidence that it is not selection
noise from one validation split. For the deployed model the paired bootstrap interval
over photographs is [-0.003, +0.016] for the mean and [-0.002, +0.019] for max x
fraction, so a single model's interval still includes zero.

The gain is larger on photographs with several annotators: for e40, 0.4712 to 0.4793
(mean) and to 0.4862 (max x fraction, floor 0.92) on those 47 photographs, against
0.4693 to 0.4731 on single-annotator photographs (`analysis/bysubset.py`). The test set
has at least 2.0 annotator records per photograph against 1.7 in validation.

Adding three shifted views to the eight (Swin-S hold-out model, the only one with
shifted views saved) changes nothing: 0.4766 to 0.4741 (mean), 0.4775 to 0.4788
(`analysis/shift_test.py`). A second model's views do not add either: e40 plus ctrl31,
sixteen views, scores 0.474 to 0.476 (`analysis/tta_multi.py`).

An earlier eight-view fusion of the same model (exp144, different clustering and score,
validation 0.4700) scored 0.40 in public, so transfer is not assured.

## What this predicts

With the offset measured by the two probes (0.059 to 0.066), the consensus candidates
land at an unrounded public score of about 0.415 to 0.418. The display flips from 0.41
to 0.42 at 0.415, so each candidate is roughly an even chance.
