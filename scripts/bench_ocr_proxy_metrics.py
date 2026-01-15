#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd


ID_RE = re.compile(r"^(?P<id>.+?)\.txt$", re.IGNORECASE)

# Some handy regexes
RE_DATE1 = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b")
RE_DATE2 = re.compile(r"\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})\b")
RE_YEAR = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
RE_DIGIT = re.compile(r"\d")
RE_ALPHA = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
RE_WEIRD = re.compile(r"[{}[\]<>]|\\x[0-9a-fA-F]{2}")
RE_MULTI_WS = re.compile(r"\s+")
RE_HYPHEN_EOL = re.compile(r"-\s*\n\s*")  # line break after hyphen


def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def normalize_for_stats(s: str) -> str:
    # light normalization only for counting
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = RE_HYPHEN_EOL.sub("", s)
    return s


@dataclass
class DocProxy:
    identifier: str
    path: str
    n_chars: int
    n_lines: int
    n_tokens: int
    pct_alpha: float
    pct_digit: float
    pct_ascii: float
    n_years: int
    n_dates: int
    weird_count: int
    empty: bool


def compute_doc_proxy(identifier: str, path: Path) -> DocProxy:
    raw = safe_read_text(path)
    s = normalize_for_stats(raw).strip("\n")
    empty = (len(s.strip()) == 0)

    n_chars = len(s)
    lines = [ln for ln in s.split("\n")]
    n_lines = len(lines) if s else 0

    tokens = [t for t in RE_MULTI_WS.split(s.strip()) if t] if s else []
    n_tokens = len(tokens)

    alpha = len(RE_ALPHA.findall(s))
    digit = len(RE_DIGIT.findall(s))
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)

    pct_alpha = (alpha / n_chars) if n_chars else 0.0
    pct_digit = (digit / n_chars) if n_chars else 0.0
    pct_ascii = (ascii_cnt / n_chars) if n_chars else 0.0

    n_years = len(RE_YEAR.findall(s))
    n_dates = len(RE_DATE1.findall(s)) + len(RE_DATE2.findall(s))

    weird_count = len(RE_WEIRD.findall(s))

    return DocProxy(
        identifier=identifier,
        path=str(path),
        n_chars=n_chars,
        n_lines=n_lines,
        n_tokens=n_tokens,
        pct_alpha=pct_alpha,
        pct_digit=pct_digit,
        pct_ascii=pct_ascii,
        n_years=n_years,
        n_dates=n_dates,
        weird_count=weird_count,
        empty=empty,
    )


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"n_docs": 0}

    def mean(x: pd.Series) -> float:
        return float(x.mean()) if len(x) else 0.0

    out: Dict[str, Any] = {}
    out["n_docs"] = int(len(df))
    out["n_empty"] = int(df["empty"].sum())
    out["pct_non_empty"] = float(100.0 * (1.0 - out["n_empty"] / max(out["n_docs"], 1)))

    for col in ["n_chars", "n_lines", "n_tokens", "pct_alpha", "pct_digit", "pct_ascii", "n_years", "n_dates", "weird_count"]:
        out[f"mean_{col}"] = mean(df[col])
        out[f"p50_{col}"] = float(df[col].median()) if col in df else 0.0
        out[f"p10_{col}"] = float(df[col].quantile(0.10)) if col in df else 0.0
        out[f"p90_{col}"] = float(df[col].quantile(0.90)) if col in df else 0.0

    # A simple "proxy score" (heuristic): non-empty + alpha density + presence of years/dates - weird chars.
    # This is not a truth metric, just a comparable index between runs.
    score = (
        (1.0 - df["empty"].astype(float)) * 2.0
        + df["pct_alpha"].clip(0, 1) * 2.0
        + (df["n_years"] > 0).astype(float) * 0.5
        + (df["n_dates"] > 0).astype(float) * 0.5
        - (df["weird_count"] / (df["n_chars"].clip(lower=1))).clip(0, 1) * 1.0
    )
    out["mean_proxy_score"] = float(score.mean())
    out["p50_proxy_score"] = float(score.median())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="output/ocr/<run-name>")
    ap.add_argument("--texts-subdir", type=str, default="texts")
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    texts_dir = run_dir / args.texts_subdir
    if not texts_dir.exists():
        raise FileNotFoundError(f"Missing texts dir: {texts_dir}")

    rows: List[Dict[str, Any]] = []
    for p in sorted(texts_dir.glob("*.txt")):
        m = ID_RE.match(p.name)
        if not m:
            continue
        identifier = m.group("id")
        rows.append(asdict(compute_doc_proxy(identifier, p)))

    df = pd.DataFrame(rows).sort_values("identifier")
    out_csv = Path(args.out_csv) if args.out_csv else (run_dir / "proxy_metrics.csv")
    out_json = Path(args.out_json) if args.out_json else (run_dir / "proxy_summary.json")

    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summarize(df), ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ proxy_metrics:", out_csv)
    print("✅ proxy_summary:", out_json)
    print(out_json.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
