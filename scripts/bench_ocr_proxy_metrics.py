#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# ---------------------------------------------------------------------
# Regex importantes (à conserver)
# ---------------------------------------------------------------------

# Identifiant doc (fallback)
IDENTIFIER_RE = re.compile(r"^(?P<id>.+)$")

# Dates courantes dans brevets (DE / FR / EN)
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b"),      # 12.03.1901
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),            # 1901-03-12
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),        # 12/03/1901
    re.compile(r"\b(\d{4})\b"),                        # fallback year
]


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def identifier_from_path(p: str) -> str:
    return Path(p).stem


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def extract_dates(text: str) -> List[str]:
    """Extract date-like strings using regexes."""
    dates: List[str] = []
    for pat in DATE_PATTERNS:
        dates.extend(m.group(1) for m in pat.finditer(text))
    return dates


# ---------------------------------------------------------------------
# Proxy metrics (OCR health / cheap signals)
# ---------------------------------------------------------------------
def compute_doc_proxy(identifier: str, text: str) -> Dict[str, Any]:
    lines = [l for l in text.splitlines() if l.strip()]
    dates = extract_dates(text)

    return {
        "identifier": identifier,
        "n_chars": len(text),
        "n_lines": len(lines),
        "n_empty_lines": max(len(text.splitlines()) - len(lines), 0),
        "avg_line_len": (sum(len(l) for l in lines) / len(lines)) if lines else 0.0,
        "n_dates": len(dates),
        "dates": "|".join(dates[:20]),  # cap pour éviter CSV énormes
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path, help="OCR run dir")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    texts_dir = run_dir / "texts"

    out_metrics = run_dir / "proxy_metrics.csv"
    out_summary = run_dir / "proxy_summary.json"

    report_file = run_dir / "ocr_report.csv"

    rows: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Preferred path: OCR report
    # -----------------------------------------------------------------
    if report_file.exists():
        df_report = pd.read_csv(report_file)

        for r in df_report.to_dict(orient="records"):
            identifier = identifier_from_path(
                r.get("out_txt") or r.get("file_name") or ""
            )

            status = r.get("status", "unknown")
            error = r.get("error", "")

            txt_path = (
                Path(r["out_txt"]) if isinstance(r.get("out_txt"), str) else None
            )
            txt_exists = bool(txt_path and txt_path.exists())

            base: Dict[str, Any] = {
                "identifier": identifier,
                "status": status,
                "txt_exists": txt_exists,
                "txt_path": str(txt_path) if txt_path else "",
                "ocr_error": error,
            }

            if status == "ok" and txt_exists:
                text = read_text(txt_path)
                if text.strip():
                    base.update(compute_doc_proxy(identifier, text))

            rows.append(base)

    # -----------------------------------------------------------------
    # Fallback: scan texts/ directly
    # -----------------------------------------------------------------
    elif texts_dir.exists():
        for p in sorted(texts_dir.glob("*.txt")):
            identifier = identifier_from_path(p.name)
            text = read_text(p)

            base = {
                "identifier": identifier,
                "status": "unknown",
                "txt_exists": True,
                "txt_path": str(p),
                "ocr_error": "",
            }

            if text.strip():
                base.update(compute_doc_proxy(identifier, text))

            rows.append(base)

    # -----------------------------------------------------------------
    # Write outputs (never crash)
    # -----------------------------------------------------------------
    df = pd.DataFrame(rows)

    if "identifier" in df.columns and not df.empty:
        df = df.sort_values("identifier")

    df.to_csv(out_metrics, index=False)

    summary = {
        "run_dir": str(run_dir),
        "n_docs_total": int(len(df)),
        "n_docs_ok": int((df.get("status") == "ok").sum()) if "status" in df else 0,
        "n_docs_with_text": int(df.get("txt_exists", pd.Series()).sum())
        if "txt_exists" in df
        else 0,
        "n_docs_with_dates": int((df.get("n_dates", pd.Series()) > 0).sum())
        if "n_dates" in df
        else 0,
        "metrics_file": str(out_metrics),
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
