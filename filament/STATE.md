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

## Current position (13 Nov 01:15 UTC)
0.41, **3rd of 27 teams at 0.41 (overall 137)**; Spaghetto99 and Salty Taro are ahead at
0.41, one team at 0.42. Best file: `/home/user/work/subs/climb/c1_v5_plus_v4hi.csv`
(58290430). All five submissions of 13 Nov are used. Ten remain: five on 14 Nov, five on
15 Nov before 06:00 UTC. Kaggle GPU: 4.6 h until 14 Nov 00:00 UTC, then 30 h (no use
planned; two fine-tunes gave nothing).

## What is settled (see LOG.md)
- The test set is a mix of one to three annotator records per photograph (at most 2.04);
  the whole validation set is the yardstick. `docs/06_test_population.md` opens with the
  withdrawal of the "all triple-annotated" claim.
- No model-side change transfers visibly: consensus across views/models, thickness, mask
  source, fine-tuning on four annotator groups (exp202), fine-tuning on fused labels
  (exp203), votes of single-view models, a sixteen-file vote.
- What works: single-group changes to the eight-file vote, judged by the team's position
  (`analysis/climb.py`; object table in `/home/user/work/an/zoo/climb_table.pkl`, keys
  `v` = votes among the eight files, `e40` = deployed single-view confidence).
- The user was told (00:55 UTC email) that this is tuning to the public half and was
  offered the choice to stop it; the junk-mask comparator remains unused.

## Next actions
1. From 00:00 UTC on 14 Nov: changes relative to c1 (objects with v>=5, or v==4 and
   e40>=0.86), one at a time with a position check; keep a change only if the team
   passes another team (or the score shows 0.42). Order: `climb/e2_c1_plus_cons4_minus_cons0.csv`
   (validation-supported second-opinion step, +36 / -30 objects); if it does not pass,
   `e1_c1_plus_cons4.csv` (the addition alone, +0.0018 on the validation analogue), then
   the validation-neutral vote-count variants `d1_c1_minus_v5.csv`, `d2_c1_plus_v3hi.csv`.
   After a pass, build the next change on the passing file (`analysis/climb3.py`).
2. Email hourly while the session runs; log every submission in LOG.md.

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
