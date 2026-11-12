"""exp202: rank fine-tuned checkpoints on the validation records of the annotator groups that label the
triple-annotated photographs (the population the leaderboard evidence says the test set is drawn from).

Each input is a single-view validation cache {photograph: [{score, rle}, ...]} as `train.py` writes at every
evaluation. The score is exp033's `evaluate_thresholds` (no erosion, area 100) restricted to those records.

    python -m experiments.exp_202_target_ft.src.select --annotations A --caches run/e002_cache.json ... --output sel.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.exp_033_inference.src.sweep import evaluate_thresholds, records_for

TARGET = ("0101", "0102", "0104", "0501")
CONFS = (0.8, 0.85, 0.9)


def subsets(annotations: Path) -> dict:
    records, _ = records_for(annotations)
    target = {name: [(rid, truth) for rid, truth in passes if str(rid).startswith(TARGET)]
              for name, passes in records.items()}
    target = {name: passes for name, passes in target.items() if passes}
    triple = {name: passes for name, passes in records.items() if len(passes) == 3}
    return dict(all=records, target=target, triple=triple)


def score(job: tuple) -> dict:
    cache_path, annotations = job
    cache = json.loads(Path(cache_path).read_text())
    row = dict(cache=str(cache_path))
    for label, records in subsets(Path(annotations)).items():
        usable = {name: passes for name, passes in records.items() if name in cache}
        if not usable:
            row[label] = None
            continue
        rows = evaluate_thresholds(cache, usable, CONFS, grow=0, min_area=100)
        row[label] = {str(r["conf"]): round(r["pq"], 5) for r in rows}
        row[label + "_records"] = sum(len(p) for p in usable.values())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    jobs = [(str(path), str(args.annotations)) for path in args.caches]
    with ProcessPoolExecutor(max_workers=max(1, min(args.processes, len(jobs)))) as pool:
        rows = list(pool.map(score, jobs))
    for row in rows:
        row["key"] = max(row["target"].values()) if row.get("target") else -1.0
        print("SELECT", json.dumps(row, sort_keys=True), flush=True)
    best = max(rows, key=lambda row: row["key"])
    args.output.write_text(json.dumps(dict(best=best["cache"], rows=rows), indent=1))
    print("SELECTED", best["cache"], best["key"], flush=True)


if __name__ == "__main__":
    main()
