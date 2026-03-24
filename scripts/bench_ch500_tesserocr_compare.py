#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def detect_tessdata_path(explicit: str | None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("TESSDATA_PREFIX"),
        "/opt/homebrew/share/tessdata",
        "/usr/local/share/tessdata",
        "/usr/share/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/tessdata",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return str(Path(candidate))
    return None


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_metric(value: Any, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}{suffix}"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(
        description="Run CH500 tesserocr OCR on multiple segmentation modes, then score each run against gold ocr_text."
    )
    ap.add_argument("--raw-dir", type=Path, default=repo_root / "data/gold_standard_CH/PNGs_extracted")
    ap.add_argument("--gold-jsonl", type=Path, default=repo_root / "data/gold_standard_CH/ch500_swiss_gold_manual.jsonl")
    ap.add_argument("--out-root", type=Path, default=repo_root / "output/ch_ocr")
    ap.add_argument("--run-prefix", type=str, default="tesserocr_ch500_seg")
    ap.add_argument("--modes", nargs="+", default=["custom", "backend"], choices=["custom", "backend"])
    ap.add_argument("--lang", type=str, default="deu+fra")
    ap.add_argument("--config", type=str, default="--psm 3 --oem 3 -c preserve_interword_spaces=1")
    ap.add_argument("--preprocess", type=str, default="light", choices=["none", "gray", "light"])
    ap.add_argument("--tessdata-path", type=str, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--custom-workers", type=int, default=None)
    ap.add_argument("--backend-workers", type=int, default=None)
    ap.add_argument("--parallel", type=str, default="threads", choices=["none", "threads", "processes"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timings", type=str, default="basic", choices=["off", "basic", "detailed"])
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--deskew-max-angle", type=float, default=20.0)
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tessdata_path = detect_tessdata_path(args.tessdata_path)
    backend_kwargs: Dict[str, Any] = {}
    if tessdata_path:
        backend_kwargs["tessdata_path"] = tessdata_path

    ocr_config = {
        "lang": args.lang,
        "config": args.config,
        "preprocess": args.preprocess,
    }

    args.out_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []

    for mode in args.modes:
        run_name = f"{args.run_prefix}_{mode}"
        workers = args.workers
        if mode == "custom" and args.custom_workers is not None:
            workers = args.custom_workers
        if mode == "backend" and args.backend_workers is not None:
            workers = args.backend_workers

        cmd = [
            sys.executable,
            "scripts/bench_run_ocr.py",
            "--raw-dir",
            str(args.raw_dir),
            "--out-root",
            str(args.out_root),
            "--run-name",
            run_name,
            "--segmentation",
            mode,
            "--backend",
            "tesserocr",
            "--backend-kwargs-json",
            json.dumps(backend_kwargs, ensure_ascii=False),
            "--ocr-config-json",
            json.dumps(ocr_config, ensure_ascii=False),
            "--workers",
            str(workers),
            "--parallel",
            args.parallel,
            "--timings",
            args.timings,
            "--log-every",
            str(args.log_every),
            "--deskew-max-angle",
            str(args.deskew_max_angle),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.no_deskew:
            cmd.append("--no-deskew")
        if args.force:
            cmd.append("--force")

        run_cmd(cmd, cwd=repo_root)

        score_cmd = [
            sys.executable,
            "scripts/bench_score_ocr.py",
            "--gold-jsonl",
            str(args.gold_jsonl),
            "--run-dir",
            str(args.out_root / run_name),
        ]
        run_cmd(score_cmd, cwd=repo_root)

        summary = load_json(args.out_root / run_name / "ocr_summary.json")
        summary["segmentation"] = mode
        summary["run_name"] = run_name
        summary["workers"] = workers
        summaries.append(summary)

    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        metrics = summary.get("metrics", {})
        metrics_matched = summary.get("metrics_matched", {})
        rows.append(
            {
                "segmentation": summary.get("segmentation"),
                "run_name": summary.get("run_name"),
                "workers": summary.get("workers"),
                "n_scored_docs": summary.get("n_scored_docs"),
                "n_matched_predictions": summary.get("n_matched_predictions"),
                "n_missing_predictions": summary.get("n_missing_predictions"),
                "cer_raw_mean": metrics.get("cer_raw", {}).get("mean"),
                "cer_raw_median": metrics.get("cer_raw", {}).get("median"),
                "cer_ws_mean": metrics.get("cer_ws", {}).get("mean"),
                "cer_ws_median": metrics.get("cer_ws", {}).get("median"),
                "wer_ws_mean": metrics.get("wer_ws", {}).get("mean"),
                "wer_ws_median": metrics.get("wer_ws", {}).get("median"),
                "cer_ws_mean_matched": metrics_matched.get("cer_ws", {}).get("mean"),
                "wer_ws_mean_matched": metrics_matched.get("wer_ws", {}).get("mean"),
                "similarity_ws_mean_matched": metrics_matched.get("similarity_ws", {}).get("mean"),
                "t_total_s_mean_matched": metrics_matched.get("t_total_s", {}).get("mean"),
                "similarity_ws_mean": metrics.get("similarity_ws", {}).get("mean"),
                "len_ratio_ws_mean": metrics.get("len_ratio_ws", {}).get("mean"),
                "t_ocr_s_mean": metrics.get("t_ocr_s", {}).get("mean"),
                "t_total_s_mean": metrics.get("t_total_s", {}).get("mean"),
            }
        )

    compare_dir = args.out_root / f"{args.run_prefix}_comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    compare_csv = compare_dir / "summary.csv"
    compare_json = compare_dir / "summary.json"

    df = pd.DataFrame(rows).sort_values("segmentation")
    df.to_csv(compare_csv, index=False)

    payload: Dict[str, Any] = {
        "raw_dir": str(args.raw_dir),
        "gold_jsonl": str(args.gold_jsonl),
        "out_root": str(args.out_root),
        "run_prefix": args.run_prefix,
        "tessdata_path": tessdata_path,
        "ocr_config": ocr_config,
        "backend_kwargs": backend_kwargs,
        "runs": summaries,
        "summary_csv": str(compare_csv),
    }

    if len(rows) == 2:
        by_mode = {row["segmentation"]: row for row in rows}
        if "custom" in by_mode and "backend" in by_mode:
            payload["delta_backend_minus_custom"] = {
                "cer_ws_mean": (
                    None
                    if by_mode["backend"]["cer_ws_mean"] is None or by_mode["custom"]["cer_ws_mean"] is None
                    else by_mode["backend"]["cer_ws_mean"] - by_mode["custom"]["cer_ws_mean"]
                ),
                "cer_ws_mean_matched": (
                    None
                    if by_mode["backend"]["cer_ws_mean_matched"] is None or by_mode["custom"]["cer_ws_mean_matched"] is None
                    else by_mode["backend"]["cer_ws_mean_matched"] - by_mode["custom"]["cer_ws_mean_matched"]
                ),
                "wer_ws_mean": (
                    None
                    if by_mode["backend"]["wer_ws_mean"] is None or by_mode["custom"]["wer_ws_mean"] is None
                    else by_mode["backend"]["wer_ws_mean"] - by_mode["custom"]["wer_ws_mean"]
                ),
                "wer_ws_mean_matched": (
                    None
                    if by_mode["backend"]["wer_ws_mean_matched"] is None or by_mode["custom"]["wer_ws_mean_matched"] is None
                    else by_mode["backend"]["wer_ws_mean_matched"] - by_mode["custom"]["wer_ws_mean_matched"]
                ),
                "t_total_s_mean": (
                    None
                    if by_mode["backend"]["t_total_s_mean"] is None or by_mode["custom"]["t_total_s_mean"] is None
                    else by_mode["backend"]["t_total_s_mean"] - by_mode["custom"]["t_total_s_mean"]
                ),
                "t_total_s_mean_matched": (
                    None
                    if by_mode["backend"]["t_total_s_mean_matched"] is None or by_mode["custom"]["t_total_s_mean_matched"] is None
                    else by_mode["backend"]["t_total_s_mean_matched"] - by_mode["custom"]["t_total_s_mean_matched"]
                ),
            }

    compare_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[compare] done")
    print(f"[compare] summary_csv={compare_csv}")
    print(f"[compare] summary_json={compare_json}")
    for row in rows:
        print(
            "[compare] "
            f"{row['segmentation']}: cer_ws_mean_matched={fmt_metric(row['cer_ws_mean_matched'])} "
            f"wer_ws_mean_matched={fmt_metric(row['wer_ws_mean_matched'])} "
            f"t_total_s_mean_matched={fmt_metric(row['t_total_s_mean_matched'], digits=3, suffix='s')}"
        )


if __name__ == "__main__":
    main()
