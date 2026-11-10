# Solar Filament Segmentation Challenge 2026 — working notes

Work toward a public leaderboard score of 0.42 on the Kaggle competition
`filament-segmentation-2026` (team "supersede"). This directory holds the analysis
code, the experiment log and the reasoning behind every submission made from
10 November 2026 onward. Earlier work (experiments exp001–exp201 and the
ensemble searches recorded in `../for shame.txt`) is summarised in
`docs/01_prior_state.md`.

| Path | Contents |
|---|---|
| `docs/00_rules.md` | What the competition rules and the organisers' forum replies allow and forbid |
| `docs/01_prior_state.md` | Where the project stood on 10 Nov and why earlier gains did not transfer |
| `docs/02_scoring_protocol.md` | The pooled-annotator scoring hypothesis, its evidence and the probe that tests it |
| `LOG.md` | Chronological log of every experiment and submission |
| `analysis/` | Scripts that produce every number quoted in the docs |

No competition data, model weights, or external annotations are stored here.
The public MAGFiLO release (which contains the test labels) is never downloaded or used.

## Reproducing the analysis

The scripts expect the competition data unpacked under
`/home/user/work/comp/data/MAGFiLO_1.0_Kaggle_2026` and the deployed model's candidate
caches (Kaggle notebook outputs `sol-exp147-candidate-cache-t4` and
`sol-exp151a-e040-test-cache-t4`) under `/home/user/work/caches/`. Paths are set at the
top of `analysis/lib.py` and `analysis/fast.py`.

```bash
python3 analysis/interann.py     # annotator-vs-annotator agreement
python3 analysis/valscore.py     # deployed model: per-annotator and pooled PQ
python3 analysis/sweep3.py       # confidence-floor sweep under both protocols
python3 analysis/build_sub.py 0.70 out.csv   # build and check a submission
```
