# Operational state (read this first when resuming)

Last updated: 12 Nov 2026, 15:30 UTC.

## Constraints that shape the work
- The session can be stopped by the account's five-hour usage limit; nothing runs and no
  email can be sent while limited. Background processes do not survive, files do. After
  the conversation was compacted on 12 Nov a window lasts far longer than before.
- Emails go to k.kasozi69@gmail.com every hour while the session runs: identify as Claude
  (model claude-fable-5-2, xhigh reasoning), then flowing paragraphs. Sent on 12 Nov at
  about 14:33 and 14:53 UTC (the second was wrongly labelled 16:30; say so in the next).
- Five submissions per UTC day. Competition closes 15 Nov 06:00 UTC.
- Kaggle GPU quota: 9.6 h until 14 Nov 00:00 UTC, then 30 h. exp202 uses about 3 h of it.
- Never download or use the public MAGFiLO annotations (they contain the test labels).
- No subagents or extra sessions unless the user asks.

## Current position (12 Nov 19:45 UTC)
0.41, 4th of 25 teams at 0.41 (overall 138); one team at 0.42. All five submissions of
12 Nov are used (blendW_thr50 0.41, blend41_thr50_thick40 0.41, blend41_thr60_thick40
0.41, cand_C 0.40, blend8_thr38 0.41); none moved the position. Best file:
`blend41_thr50.csv` or `blend41_thr60.csv`. Fifteen submissions remain (13, 14 and
15 Nov before 06:00 UTC). Kaggle GPU: about 7.1 h until 14 Nov 00:00 UTC, then 30 h.

## What is settled (see LOG.md, docs/06_test_population.md)
The test set is triple-annotated; the ceiling with the labels in hand is about 0.59; the
fine-tune on the test-side annotators (exp202, outputs in `/home/user/work/caches/exp202`)
gave no gain. No systematic lever is left untested. The user was asked by email (19:45
UTC) whether to use a junk-mask leaderboard comparator; do not use it without their yes.

## Next actions
1. From 00:00 UTC on 13 Nov, submit one at a time from `/home/user/work/subs/` with a
   position check after each (see `docs/08_candidate_pool.txt` for validation scores and
   mutual agreement): `voteS9_k4.csv`, `pool13/five85.csv`,
   `deployed_sel_three_masks.csv`, `pool13/d4_swinb.csv`, `pool13/S6noe40_k3.csv`.
   If one passes a team, stay in its family for the next slot; avoid the max-times-fraction
   consensus rule (two of two below 0.405).
2. 14 and 15 Nov (ten more slots): other families from the pool that are least like the
   files already submitted (`pool13/V5d4_k3.csv`, `pool13/d4_ftB4.csv`,
   `pool13/d4_swins.csv`, `pool13/swinsswinb85.csv`, `blend16_thr50.csv`, ...).
3. Email hourly while the session runs; log every submission in LOG.md.
4. The junk-mask comparator stays unused unless the user says yes (asked by email 19:45).

## Where things are
- Kaggle CLI: `export KAGGLE_USERNAME=cosmicwitness KAGGLE_KEY=placeholder` (the proxy
  supplies the real credential). Competition id `filament-segmentation-2026`.
- Working directory (not in git): `/home/user/work/` with `comp/data` (competition data),
  `caches/` (model prediction caches from Kaggle notebook outputs), `an/` (scripts, same
  as `filament/analysis/`, plus pickles under `an/zoo/`), `subs/` (submission files).
- Candidate files in `/home/user/work/subs/`: `cand_A..G_*.csv` (see
  `docs/04_candidates.md`). Rebuild with `python3 make_d4_sub.py <mean8|maxfrac> <floor>
  <out>` or `python3 make_multi_sub.py e40,swins,swinb <mean8|maxfrac> <floor> <out>`
  from `/home/user/work/an`; both replay validation before writing.
- Leaderboard position: download
  `https://www.kaggle.com/api/v1/competitions/filament-segmentation-2026/leaderboard/download`
  into its own directory and read it with a script kept elsewhere. Team "supersede",
  5th of 19 at 0.41 on 10 Nov; one team at 0.42.

## What is established
1. Leaderboard = per-annotator PQ (organisers' notebook). Public score of the deployed
   single-view model = validation PQ minus 0.059..0.066. Pooled-annotator scoring and a
   sparser test set were both tested by probe submissions and rejected (0.39, 0.40).
2. Models are at a label-noise ceiling: annotators agree at PQ 0.34; selection using other
   annotators' real masks (0.447) is worse than the model's own confidence (0.471);
   six-model consensus 0.4701 = single model 0.4705.
3. Eight-view (D4) consensus of one model gains 0.005..0.019 on all six models tested.
   Three strongest models, eight views each: 0.4817 / 0.4824 on validation (twins),
   +0.011 with bootstrap interval [+0.003, +0.021].
4. Dead ends: 2560-px view, shifted views on top of D4, other models as extra views,
   decoder-layer and mask-sharpness scores, mask threshold choice, painting policy,
   learned rankers (no better than the simple rules).

## Leaderboard results on 11 Nov
G 0.40, F 0.41 (no change in position), blend41_thr50 0.41 and blend41_thr60 0.41 (team
moved 5th -> 3rd of 20 at 0.41), blend11_thr50 0.41 (no change). The eight-view consensus
alone does not transfer; a majority-vote consensus of distinct 0.41 submissions does
improve the unrounded score. Usage windows are about 45 calls each; the limit pauses the
session for the rest of each five-hour window.

## Next actions for 12 Nov (slots reopen 00:00 UTC)
Submit one at a time, check score and position after each (two teams, Salty Taro and
Spaghetto99, are ahead at 0.41, then 0.42): `/home/user/work/subs/blendW_thr50.csv`,
`blend8_thr38.csv`, `blendW_thr40.csv`; then variants decided from the results (for
example the eight 0.41 files plus the YOLO files at full weight). Rebuild with
`python3 blendw.py` in `/home/user/work/an` (inputs under `/home/user/work/subs/lb41`,
`lb40`, `lb39`, downloaded with `kaggle competitions submission-download <id>`).

## Older next actions (done)
1. After 00:20 UTC on 11 Nov: confirm no submission has been made that day, submit
   `cand_G_three_maxfrac_092.csv`, read score and rank within the 0.41 group, then
   `cand_F_three_mean8_088.csv`, then `cand_C_e40_d4_maxfrac_092.csv`; decide slots 4-5
   from the results. Log each in `LOG.md`.
2. If 0.42 appears: record it, tell the user, and note which file to select as final.
3. If not: variants on the flat part of the floor curve (G at 0.94, F at 0.85), and D/E.
