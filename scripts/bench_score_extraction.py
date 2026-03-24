#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------
def load_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-jsonl", required=True, type=Path)
    ap.add_argument("--pred-jsonl", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = out_dir / "scores.csv"
    summary_json = out_dir / "summary.json"

    gold = load_jsonl(args.gold_jsonl)
    pred = load_jsonl(args.pred_jsonl)

    # -----------------------------------------------------------------
    # Empty cases (do not crash the bench)
    # -----------------------------------------------------------------
    if not gold or not pred:
        pd.DataFrame([]).to_csv(scores_csv, index=False)

        summary = {
            "status": "empty",
            "reason": "gold_empty" if not gold else "pred_empty",
            "gold_docs": int(len(gold)),
            "pred_docs": int(len(pred)),
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[score] Empty input – skipping scoring", file=sys.stderr)
        return

    gold_by_id = {x["identifier"]: x for x in gold if "identifier" in x}
    pred_by_id = {x["identifier"]: x for x in pred if "identifier" in x}

    common_ids = sorted(set(gold_by_id) & set(pred_by_id))

    if not common_ids:
        pd.DataFrame([]).to_csv(scores_csv, index=False)

        summary = {
            "status": "empty",
            "reason": "no_common_identifiers",
            "gold_docs": int(len(gold_by_id)),
            "pred_docs": int(len(pred_by_id)),
            "common_docs": 0,
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[score] No common identifiers", file=sys.stderr)
        return

    # -----------------------------------------------------------------
    # Minimal scoring (placeholder, extensible)
    # -----------------------------------------------------------------
    rows: List[Dict] = []

    for i in common_ids:
        g = gold_by_id[i]
        p = pred_by_id[i]

        rows.append(
            {
                "identifier": i,
                "gold_len": len(g.get("text", "")),
                "pred_len": len(p.get("text", "")),
                "len_diff": abs(len(g.get("text", "")) - len(p.get("text", ""))),
            }
        )

    df = pd.DataFrame(rows).sort_values("identifier")
    df.to_csv(scores_csv, index=False)

    summary = {
        "status": "ok",
        "gold_docs": int(len(gold_by_id)),
        "pred_docs": int(len(pred_by_id)),
        "common_docs": int(len(common_ids)),
        "scores_file": str(scores_csv),
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
