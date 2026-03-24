#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List

import pandas as pd
from rapidfuzz.distance import Levenshtein


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_identifier(raw: str) -> str:
    stem = Path(raw or "").stem.strip()
    if stem.endswith("_full"):
        stem = stem[: -len("_full")]
    return stem


def normalize_ws(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num) / float(den)


def aggregate_metric(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, float | None]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def load_predictions(texts_dir: Path) -> Dict[str, Path]:
    preds: Dict[str, Path] = {}
    if not texts_dir.exists():
        return preds
    for txt_path in sorted(texts_dir.glob("*.txt")):
        preds[normalize_identifier(txt_path.name)] = txt_path
    return preds


def load_report_by_id(report_path: Path) -> Dict[str, Dict[str, Any]]:
    if not report_path.exists():
        return {}
    df = pd.read_csv(report_path)
    rows: Dict[str, Dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        identifier = normalize_identifier(
            str(row.get("out_txt") or row.get("file_name") or "")
        )
        rows[identifier] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score OCR texts from a run dir against a gold JSONL field."
    )
    ap.add_argument("--gold-jsonl", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--gold-id-key", type=str, default="identifier")
    ap.add_argument("--gold-file-key", type=str, default="file_name")
    ap.add_argument("--gold-text-key", type=str, default="ocr_text")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    out_dir = args.out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    texts_dir = run_dir / "texts"
    report_by_id = load_report_by_id(run_dir / "ocr_report.csv")
    pred_paths = load_predictions(texts_dir)
    gold_rows = load_jsonl(args.gold_jsonl)

    score_rows: List[Dict[str, Any]] = []

    for gold in gold_rows:
        identifier = str(gold.get(args.gold_id_key) or "").strip()
        if not identifier:
            identifier = normalize_identifier(str(gold.get(args.gold_file_key) or ""))
        if not identifier:
            continue

        gold_text = str(gold.get(args.gold_text_key) or "")
        pred_path = pred_paths.get(identifier)
        pred_text = ""
        if pred_path and pred_path.exists():
            pred_text = pred_path.read_text(encoding="utf-8", errors="replace")

        gold_ws = normalize_ws(gold_text)
        pred_ws = normalize_ws(pred_text)
        gold_words = gold_ws.split()
        pred_words = pred_ws.split()

        char_dist_raw = Levenshtein.distance(pred_text, gold_text)
        char_dist_ws = Levenshtein.distance(pred_ws, gold_ws)
        word_dist_ws = Levenshtein.distance(pred_words, gold_words)

        report = report_by_id.get(identifier, {})
        score_rows.append(
            {
                "identifier": identifier,
                "gold_file_name": gold.get(args.gold_file_key, ""),
                "pred_path": str(pred_path) if pred_path else "",
                "pred_missing": pred_path is None,
                "status": report.get("status", "missing"),
                "ocr_error": report.get("error", ""),
                "gold_chars_raw": len(gold_text),
                "pred_chars_raw": len(pred_text),
                "gold_chars_ws": len(gold_ws),
                "pred_chars_ws": len(pred_ws),
                "gold_words_ws": len(gold_words),
                "pred_words_ws": len(pred_words),
                "char_dist_raw": char_dist_raw,
                "char_dist_ws": char_dist_ws,
                "word_dist_ws": word_dist_ws,
                "cer_raw": safe_div(char_dist_raw, len(gold_text)),
                "cer_ws": safe_div(char_dist_ws, len(gold_ws)),
                "wer_ws": safe_div(word_dist_ws, len(gold_words)),
                "len_ratio_raw": safe_div(len(pred_text), len(gold_text)),
                "len_ratio_ws": safe_div(len(pred_ws), len(gold_ws)),
                "similarity_ws": 1.0 - safe_div(char_dist_ws, max(len(gold_ws), len(pred_ws), 1)),
                "n_blocks": report.get("n_blocks"),
                "n_blocks_kept": report.get("n_blocks_kept"),
                "deskew_angle": report.get("deskew_angle"),
                "t_ocr_s": report.get("t_ocr_s"),
                "t_total_s": report.get("t_total_s"),
            }
        )

    df = pd.DataFrame(score_rows)
    if not df.empty and "identifier" in df.columns:
        df = df.sort_values("identifier")
    scores_csv = out_dir / "ocr_scores.csv"
    summary_json = out_dir / "ocr_summary.json"
    worst_csv = out_dir / "ocr_worst20.csv"

    df.to_csv(scores_csv, index=False)
    if not df.empty:
        df.sort_values(["cer_ws", "cer_raw", "identifier"], ascending=[False, False, True]).head(20).to_csv(
            worst_csv, index=False
        )
    else:
        pd.DataFrame([]).to_csv(worst_csv, index=False)

    matched_rows = [row for row in score_rows if not row["pred_missing"]]

    summary = {
        "status": "ok",
        "run_dir": str(run_dir),
        "texts_dir": str(texts_dir),
        "gold_jsonl": str(args.gold_jsonl),
        "gold_id_key": args.gold_id_key,
        "gold_text_key": args.gold_text_key,
        "n_gold_rows": int(len(gold_rows)),
        "n_scored_docs": int(len(df)),
        "n_pred_docs": int(len(pred_paths)),
        "n_matched_predictions": int(len(matched_rows)),
        "n_missing_predictions": int(df["pred_missing"].sum()) if not df.empty else 0,
        "n_ok_status": int((df["status"] == "ok").sum()) if "status" in df else 0,
        "coverage": safe_div(len(matched_rows), len(df)),
        "metrics": {
            "cer_raw": aggregate_metric(score_rows, "cer_raw"),
            "cer_ws": aggregate_metric(score_rows, "cer_ws"),
            "wer_ws": aggregate_metric(score_rows, "wer_ws"),
            "len_ratio_raw": aggregate_metric(score_rows, "len_ratio_raw"),
            "len_ratio_ws": aggregate_metric(score_rows, "len_ratio_ws"),
            "similarity_ws": aggregate_metric(score_rows, "similarity_ws"),
            "t_ocr_s": aggregate_metric(score_rows, "t_ocr_s"),
            "t_total_s": aggregate_metric(score_rows, "t_total_s"),
        },
        "metrics_matched": {
            "cer_raw": aggregate_metric(matched_rows, "cer_raw"),
            "cer_ws": aggregate_metric(matched_rows, "cer_ws"),
            "wer_ws": aggregate_metric(matched_rows, "wer_ws"),
            "len_ratio_raw": aggregate_metric(matched_rows, "len_ratio_raw"),
            "len_ratio_ws": aggregate_metric(matched_rows, "len_ratio_ws"),
            "similarity_ws": aggregate_metric(matched_rows, "similarity_ws"),
            "t_ocr_s": aggregate_metric(matched_rows, "t_ocr_s"),
            "t_total_s": aggregate_metric(matched_rows, "t_total_s"),
        },
        "scores_csv": str(scores_csv),
        "worst_csv": str(worst_csv),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ocr-score] done")
    print(f"[ocr-score] run_dir={run_dir}")
    print(
        "[ocr-score] "
        f"scored_docs={summary['n_scored_docs']} "
        f"matched_predictions={summary['n_matched_predictions']} "
        f"missing_predictions={summary['n_missing_predictions']}"
    )
    print(
        "[ocr-score] cer_ws_mean_all="
        f"{summary['metrics']['cer_ws']['mean']:.4f}" if summary["metrics"]["cer_ws"]["mean"] is not None else "[ocr-score] cer_ws_mean_all=NA"
    )
    print(
        "[ocr-score] cer_ws_mean_matched="
        f"{summary['metrics_matched']['cer_ws']['mean']:.4f}"
        if summary["metrics_matched"]["cer_ws"]["mean"] is not None
        else "[ocr-score] cer_ws_mean_matched=NA"
    )
    print(f"[ocr-score] scores_csv={scores_csv}")
    print(f"[ocr-score] summary_json={summary_json}")


if __name__ == "__main__":
    main()
