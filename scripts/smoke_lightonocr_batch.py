#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


def _run_bench(
    *,
    raw_dir: Path,
    out_root: Path,
    run_name: str,
    batch_size: int,
    limit: int,
    backend_kwargs_json: str,
    base_ocr_config: Dict[str, object],
) -> Path:
    cfg = dict(base_ocr_config)
    cfg["batch_size"] = int(batch_size)
    cmd = [
        sys.executable,
        "scripts/bench_run_ocr.py",
        "--raw-dir",
        str(raw_dir),
        "--out-root",
        str(out_root),
        "--run-name",
        run_name,
        "--backend",
        "lightonocr",
        "--segmentation",
        "backend",
        "--workers",
        "1",
        "--parallel",
        "none",
        "--timings",
        "detailed",
        "--limit",
        str(limit),
        "--force",
        "--backend-kwargs-json",
        backend_kwargs_json,
        "--ocr-config-json",
        json.dumps(cfg, ensure_ascii=False),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_root / run_name / "ocr_report.csv"


def _read_report(path: Path) -> Tuple[int, int]:
    rows = 0
    errors = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            if row.get("status") == "error":
                errors += 1
    return rows, errors


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Smoke test LightOnOCR batching: compare batch_size=1 vs batch_size=4."
    )
    ap.add_argument("--raw-dir", type=Path, required=True, help="Input image/pdf directory.")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root for smoke runs.")
    ap.add_argument("--run-prefix", type=str, default="smoke_lightonocr_batch")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument(
        "--backend-kwargs-json",
        type=str,
        default="{}",
        help='JSON for LightOnOcrBackend constructor (e.g. {"auto_install_deps": true}).',
    )
    ap.add_argument(
        "--ocr-config-json",
        type=str,
        default='{"max_new_tokens":4096,"resize_longest_edge":1540,"temperature":0.0,"do_sample":false}',
        help="Base OCR config JSON; script overrides batch_size.",
    )
    args = ap.parse_args()

    base_ocr_config = json.loads(args.ocr_config_json or "{}")
    run1 = f"{args.run_prefix}_b1"
    run4 = f"{args.run_prefix}_b4"

    report1 = _run_bench(
        raw_dir=args.raw_dir,
        out_root=args.out_root,
        run_name=run1,
        batch_size=1,
        limit=args.limit,
        backend_kwargs_json=args.backend_kwargs_json,
        base_ocr_config=base_ocr_config,
    )
    report4 = _run_bench(
        raw_dir=args.raw_dir,
        out_root=args.out_root,
        run_name=run4,
        batch_size=4,
        limit=args.limit,
        backend_kwargs_json=args.backend_kwargs_json,
        base_ocr_config=base_ocr_config,
    )

    rows1, err1 = _read_report(report1)
    rows4, err4 = _read_report(report4)
    print(f"batch=1: rows={rows1} errors={err1} report={report1}")
    print(f"batch=4: rows={rows4} errors={err4} report={report4}")

    if rows1 != rows4:
        raise SystemExit(f"Mismatch in output rows: batch1={rows1} vs batch4={rows4}")
    if err1 != 0 or err4 != 0:
        raise SystemExit(f"Errors found in reports: batch1={err1}, batch4={err4}")

    print("OK: same number of outputs, no report errors.")


if __name__ == "__main__":
    main()
