#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_patent_pipeline")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rapidfuzz import fuzz, process


MONTH_VARIANTS = {
    1: ["janvier", "janv", "januar", "janner", "january", "gennaio", "jan"],
    2: ["fevrier", "fevr", "februar", "february", "febbraio", "feb"],
    3: ["mars", "marz", "march", "marzo", "mar"],
    4: ["avril", "april", "aprile", "apr"],
    5: ["mai", "may", "maggio"],
    6: ["juin", "juni", "june", "giugno", "jun"],
    7: ["juillet", "juli", "july", "luglio", "jul"],
    8: ["aout", "august", "agosto", "aug"],
    9: ["septembre", "sept", "september", "settembre", "sep"],
    10: ["octobre", "oktober", "october", "ottobre", "oct", "okt"],
    11: ["novembre", "november", "nov"],
    12: ["decembre", "dezember", "december", "dicembre", "dec", "dez"],
}

FIELD_SPECS: List[Tuple[str, str]] = [
    ("title", "Titre"),
    ("inventor_name", "Inventeur nom"),
    ("inventor_address", "Inventeur adresse"),
    ("assignee_name", "Assignee nom"),
    ("assignee_address", "Assignee adresse"),
    ("pub_date_application", "Date depot"),
    ("pub_date_publication", "Date publication"),
    ("pub_date_foreign", "Date priorite"),
    ("classification", "Classification"),
    ("industrial_field", "Domaine industriel"),
]


@dataclass
class OcrDoc:
    identifier: str
    raw_text: str
    norm_text: str
    tokens_unique: List[str]
    token_set: set[str]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(ch)
    )


def normalize_text(text: str) -> str:
    text = strip_accents(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    out: List[str] = []
    for tok in normalize_text(text).split():
        if any(ch.isdigit() for ch in tok):
            out.append(tok)
        elif len(tok) >= 3:
            out.append(tok)
    return out


def load_ocr_run(run_dir: Path) -> Dict[str, OcrDoc]:
    docs: Dict[str, OcrDoc] = {}
    texts_dir = run_dir / "texts"
    for path in sorted(texts_dir.glob("*.txt")):
        identifier = path.stem
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        norm_text = normalize_text(raw_text)
        tokens_unique = sorted(set(tokenize(raw_text)))
        docs[identifier] = OcrDoc(
            identifier=identifier,
            raw_text=raw_text,
            norm_text=norm_text,
            tokens_unique=tokens_unique,
            token_set=set(tokens_unique),
        )
    return docs


def fuzzy_token_present(token: str, doc: OcrDoc) -> bool:
    if token in doc.token_set:
        return True
    if len(token) < 4 or not doc.tokens_unique:
        return False
    if process.extractOne(token, doc.tokens_unique, scorer=fuzz.ratio, score_cutoff=88):
        return True
    return False


def phrase_match_score(phrase: str, doc: OcrDoc) -> Optional[float]:
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return None
    if phrase_norm in doc.norm_text:
        return 1.0

    phrase_tokens = tokenize(phrase)
    if not phrase_tokens:
        return None

    matched = sum(1 for token in phrase_tokens if fuzzy_token_present(token, doc))
    return matched / len(phrase_tokens)


def phrase_is_present(phrase: str, doc: OcrDoc) -> Optional[bool]:
    score = phrase_match_score(phrase, doc)
    if score is None:
        return None
    n_tokens = len(tokenize(phrase))
    if n_tokens <= 1:
        return score >= 1.0
    if n_tokens <= 3:
        return score >= (2.0 / 3.0)
    return score >= 0.75


def date_candidates(date_iso: str) -> List[str]:
    year_s, month_s, day_s = date_iso.split("-")
    year = int(year_s)
    month = int(month_s)
    day = int(day_s)
    out = {
        f"{year_s} {month_s} {day_s}",
        f"{day_s} {month_s} {year_s}",
        f"{day} {month} {year}",
        f"{day_s}.{month_s}.{year_s}",
        f"{day}.{month}.{year}",
        f"{day_s}/{month_s}/{year_s}",
        f"{day}/{month}/{year}",
    }
    for month_name in MONTH_VARIANTS.get(month, []):
        out.add(f"{day} {month_name} {year}")
        out.add(f"{day_s} {month_name} {year}")
    return sorted(normalize_text(x) for x in out if x)


def date_is_present(date_iso: str, doc: OcrDoc) -> Optional[bool]:
    if not date_iso:
        return None
    try:
        year_s, month_s, day_s = date_iso.split("-")
    except ValueError:
        return None

    for candidate in date_candidates(date_iso):
        if candidate and candidate in doc.norm_text:
            return True

    y = int(year_s)
    m = int(month_s)
    d = int(day_s)
    raw = strip_accents(doc.raw_text).lower()
    numeric_patterns = [
        rf"\b{d:02d}[./-]{m:02d}[./-]{y}\b",
        rf"\b{d}[./-]{m}[./-]{y}\b",
        rf"\b{y}[./-]{m:02d}[./-]{d:02d}\b",
    ]
    for pat in numeric_patterns:
        if re.search(pat, raw):
            return True
    return False


def classification_is_present(value: str, doc: OcrDoc) -> Optional[bool]:
    value_norm = normalize_text(value)
    if not value_norm:
        return None
    if value_norm in doc.norm_text:
        return True

    escaped = re.escape(value_norm)
    cue_pats = [
        rf"\b(?:klasse|classe|klassierung|classification|internationale klassifikation|int cl)\b.{0,20}\b{escaped}\b",
    ]
    for pat in cue_pats:
        if re.search(pat, doc.norm_text):
            return True
    return False


def field_items(row: Dict[str, Any], field_key: str) -> List[str]:
    if field_key == "title":
        return [row.get("title", "")]
    if field_key == "inventor_name":
        return [x.get("name", "") for x in row.get("inventors", [])]
    if field_key == "inventor_address":
        return [x.get("address", "") for x in row.get("inventors", [])]
    if field_key == "assignee_name":
        return [x.get("name", "") for x in row.get("assignees", [])]
    if field_key == "assignee_address":
        return [x.get("address", "") for x in row.get("assignees", [])]
    if field_key in {"pub_date_application", "pub_date_publication", "pub_date_foreign"}:
        return [row.get(field_key, "")]
    if field_key == "classification":
        return [row.get("classification", "")]
    if field_key == "industrial_field":
        return [row.get("industrial_field", "")]
    raise KeyError(field_key)


def feature_present(field_key: str, item_value: str, doc: OcrDoc) -> Optional[bool]:
    if not (item_value or "").strip():
        return None
    if field_key.startswith("pub_date_"):
        return date_is_present(item_value, doc)
    if field_key == "classification":
        return classification_is_present(item_value, doc)
    return phrase_is_present(item_value, doc)


def feature_match_score(field_key: str, item_value: str, doc: OcrDoc) -> Optional[float]:
    if not (item_value or "").strip():
        return None
    if field_key.startswith("pub_date_"):
        present = date_is_present(item_value, doc)
        return None if present is None else float(present)
    if field_key == "classification":
        present = classification_is_present(item_value, doc)
        return None if present is None else float(present)
    return phrase_match_score(item_value, doc)


def compute_presence(
    gold_rows: Sequence[Dict[str, Any]],
    run_docs: Dict[str, OcrDoc],
    run_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    field_rows: List[Dict[str, Any]] = []
    doc_rows: List[Dict[str, Any]] = []

    for row in gold_rows:
        identifier = row["identifier"]
        doc = run_docs.get(identifier)
        if doc is None:
            continue

        total_items = 0
        matched_items = 0
        total_score = 0.0

        for field_key, _label in FIELD_SPECS:
            items = [x for x in field_items(row, field_key) if (x or "").strip()]
            matched = 0
            considered = 0
            field_score_sum = 0.0
            for item in items:
                score = feature_match_score(field_key, item, doc)
                if score is None:
                    continue
                present = feature_present(field_key, item, doc)
                considered += 1
                total_items += 1
                total_score += score
                field_score_sum += score
                if present:
                    matched += 1
                    matched_items += 1

            field_rows.append(
                {
                    "run": run_label,
                    "identifier": identifier,
                    "field_key": field_key,
                    "n_gold_items": considered,
                    "n_present_items": matched,
                    "presence_rate": (matched / considered) if considered else math.nan,
                    "match_score_sum": field_score_sum,
                    "avg_match_score": (field_score_sum / considered) if considered else math.nan,
                }
            )

        doc_rows.append(
            {
                "run": run_label,
                "identifier": identifier,
                "total_gold_items": total_items,
                "matched_items": matched_items,
                "doc_presence_score": (matched_items / total_items) if total_items else math.nan,
                "doc_match_score": (total_score / total_items) if total_items else math.nan,
            }
        )

    return pd.DataFrame(field_rows), pd.DataFrame(doc_rows)


def plot_curves(
    field_summary: pd.DataFrame,
    doc_scores: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    label_map = {key: label for key, label in FIELD_SPECS}
    colors = {
        "custom": "#b85c38",
        "backend": "#0f6cbd",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    ax1, ax2 = axes

    field_order = [key for key, _ in FIELD_SPECS if key in set(field_summary["field_key"])]
    x = list(range(len(field_order)))

    for run_label, sub in field_summary.groupby("run"):
        sub = sub.set_index("field_key").reindex(field_order)
        y = [100.0 * float(v) for v in sub["avg_match_score"]]
        ax1.plot(
            x,
            y,
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=colors.get(run_label, "#333333"),
            label=run_label,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([label_map[k] for k in field_order], rotation=35, ha="right")
    ax1.set_ylim(0, 102)
    ax1.set_ylabel("Recouvrement moyen du gold (%)")
    ax1.set_title("Qualite de presence par feature")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False)

    for run_label, sub in doc_scores.groupby("run"):
        vals = sorted(v for v in sub["doc_match_score"].tolist() if not math.isnan(v))
        if not vals:
            continue
        y = [(i + 1) / len(vals) for i in range(len(vals))]
        ax2.step(
            vals,
            y,
            where="post",
            linewidth=2.4,
            color=colors.get(run_label, "#333333"),
            label=run_label,
        )

    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Score moyen de recouvrement par document")
    ax2.set_ylabel("Part cumulative des documents")
    ax2.set_title("Courbe cumulative par document")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)

    fig.suptitle(title, fontsize=15)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot feature-presence curves for CH500 OCR runs."
    )
    ap.add_argument("--gold-jsonl", required=True, type=Path)
    ap.add_argument("--custom-run-dir", required=True, type=Path)
    ap.add_argument("--backend-run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--title", type=str, default="CH500 - presence des features dans l'OCR")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold_rows = load_jsonl(args.gold_jsonl)
    custom_docs = load_ocr_run(args.custom_run_dir)
    backend_docs = load_ocr_run(args.backend_run_dir)

    field_custom, doc_custom = compute_presence(gold_rows, custom_docs, "custom")
    field_backend, doc_backend = compute_presence(gold_rows, backend_docs, "backend")

    field_df = pd.concat([field_custom, field_backend], ignore_index=True)
    doc_df = pd.concat([doc_custom, doc_backend], ignore_index=True)

    field_summary = (
        field_df.groupby(["run", "field_key"], as_index=False)
        .agg(
            n_gold_items=("n_gold_items", "sum"),
            n_present_items=("n_present_items", "sum"),
            match_score_sum=("match_score_sum", "sum"),
        )
    )
    field_summary["presence_rate"] = field_summary["n_present_items"] / field_summary["n_gold_items"]
    field_summary["avg_match_score"] = field_summary["match_score_sum"] / field_summary["n_gold_items"]
    field_summary["field_label"] = field_summary["field_key"].map(dict(FIELD_SPECS))

    doc_summary = doc_df.copy()

    field_csv = args.out_dir / "feature_presence_by_field.csv"
    doc_csv = args.out_dir / "feature_presence_by_doc.csv"
    plot_png = args.out_dir / "feature_presence_curves.png"
    summary_json = args.out_dir / "feature_presence_summary.json"

    field_summary.sort_values(["field_key", "run"]).to_csv(field_csv, index=False)
    doc_summary.sort_values(["run", "identifier"]).to_csv(doc_csv, index=False)
    plot_curves(field_summary, doc_summary, plot_png, title=args.title)

    payload: Dict[str, Any] = {
        "gold_jsonl": str(args.gold_jsonl),
        "custom_run_dir": str(args.custom_run_dir),
        "backend_run_dir": str(args.backend_run_dir),
        "field_csv": str(field_csv),
        "doc_csv": str(doc_csv),
        "plot_png": str(plot_png),
        "field_summary": field_summary.sort_values(["field_key", "run"]).to_dict(orient="records"),
        "doc_presence_mean": (
            doc_summary.groupby("run")["doc_presence_score"].mean().reset_index().to_dict(orient="records")
        ),
        "doc_match_mean": (
            doc_summary.groupby("run")["doc_match_score"].mean().reset_index().to_dict(orient="records")
        ),
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[feature-plot] field_csv={field_csv}")
    print(f"[feature-plot] doc_csv={doc_csv}")
    print(f"[feature-plot] plot_png={plot_png}")
    print(f"[feature-plot] summary_json={summary_json}")


if __name__ == "__main__":
    main()
