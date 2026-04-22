#!/usr/bin/env python3
"""Generate a submission JSON from human annotation xlsx.

Uses the same scoring/classification logic as the VLM evaluator:
  - Three annotator scores averaged per dimension
  - is_poisoned = max(semantic, logical, decision) >= 0.6
  - attack_level = highest-scoring dimension (tie-break: Decision > Semantic > Logical)
  - final_score = 0.3 * semantic + 0.3 * logical + 0.4 * decision
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import openpyxl

POISON_THRESHOLD = 0.6
WEIGHTS = {"semantic": 0.3, "logical": 0.3, "decision": 0.4}


def compute_final_score(sem: float, log: float, dec: float) -> dict:
    final = WEIGHTS["semantic"] * sem + WEIGHTS["logical"] * log + WEIGHTS["decision"] * dec
    is_poisoned = max(sem, log, dec) >= POISON_THRESHOLD
    if not is_poisoned:
        attack_level = "None"
    else:
        ranked = sorted(
            [("Decision", dec), ("Semantic", sem), ("Logical", log)],
            key=lambda x: x[1],
            reverse=True,
        )
        top_score = ranked[0][1]
        priority = {"Decision": 0, "Semantic": 1, "Logical": 2}
        candidates = [name for name, s in ranked if s == top_score]
        attack_level = min(candidates, key=lambda n: priority[n])
    return {
        "is_poisoned": is_poisoned,
        "attack_level": attack_level,
        "scores": {
            "semantic": round(sem, 2),
            "logical": round(log, 2),
            "decision": round(dec, 2),
        },
        "final_score": round(final, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert human annotation xlsx to submission JSON")
    parser.add_argument("--annotation_xlsx", type=str,
                        default="human_evaluate/annotation.xlsx")
    parser.add_argument("--output", type=str,
                        default="results/human_submission.json")
    args = parser.parse_args()

    wb = openpyxl.load_workbook(args.annotation_xlsx)
    ws = wb.active

    results = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        idx = int(row[0])
        video_id = f"{idx:02d}.mp4"

        sem = round((float(row[1]) + float(row[4]) + float(row[7])) / 3, 2)
        log = round((float(row[2]) + float(row[5]) + float(row[8])) / 3, 2)
        dec = round((float(row[3]) + float(row[6]) + float(row[9])) / 3, 2)

        entry = compute_final_score(sem, log, dec)
        results.append({
            "video_id": video_id,
            "is_poisoned": entry["is_poisoned"],
            "attack_level": entry["attack_level"],
            "scores": entry["scores"],
            "final_score": entry["final_score"],
        })

    results.sort(key=lambda x: x["video_id"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    poisoned = sum(1 for r in results if r["is_poisoned"])
    print(f"Wrote {len(results)} entries to {args.output}")
    print(f"Poisoned: {poisoned}/{len(results)} ({poisoned/len(results)*100:.0f}%)")


if __name__ == "__main__":
    main()
