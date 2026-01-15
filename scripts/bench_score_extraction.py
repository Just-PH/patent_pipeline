#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# -----------------------------
# Optional fast fuzzy library
# -----------------------------
try:
    from rapidfuzz import fuzz  # type: ignore

    def _token_set_ratio(a: str, b: str) -> float:
        return float(fuzz.token_set_ratio(a, b)) / 100.0

except Exception:
    # Fallback (slower/less robust than rapidfuzz, but avoids hard dependency)
    import difflib

    def _token_set_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()


# -----------------------------
# Config / regex
# -----------------------------
ID_KEYS = ["identifier", "doc_id", "id", "file_name", "filename"]

RE_MULTI_WS = re.compile(r"\s+")
RE_PUNCT = re.compile(r"[^\w\sÀ-ÖØ-öø-ÿ-]+", re.UNICODE)

# Dates
RE_YMD = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")
RE_DMY = re.compile(r"^\s*(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\s*$")


# -----------------------------
# IO helpers
# -----------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def normalize_id(s: str) -> str:
    s = s.strip()
    # basename if it's a path
    s = s.split("/")[-1].split("\\")[-1]
    # strip common extensions
    for ext in (".txt", ".png", ".pdf", ".jpg", ".jpeg", ".tif", ".tiff"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    return s.strip()


def get_id(obj: Dict[str, Any]) -> Optional[str]:
    for k in ID_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return normalize_id(v)

    # common nesting patterns
    meta = obj.get("meta")
    if isinstance(meta, dict):
        for k in ID_KEYS:
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return normalize_id(v)

    return None


# -----------------------------
# Normalization
# -----------------------------
def norm_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    s = RE_MULTI_WS.sub(" ", s)
    s = RE_PUNCT.sub("", s)
    return s.lower().strip()


def norm_date(s: Any) -> Optional[str]:
    """Normalize to YYYY-MM-DD if possible; else None."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        s = str(s)
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    m = RE_YMD.match(s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = RE_DMY.match(s)
    if m:
        d = int(m.group(1))
        mo = int(m.group(2))
        y = int(m.group(3))
        if y < 100:
            # heuristic: 00-49 => 2000+, 50-99 => 1900+
            y = 1900 + y if y >= 50 else 2000 + y
        return f"{y:04d}-{mo:02d}-{d:02d}"

    # fallback: keep digits and '-' and try again
    digits = re.sub(r"[^\d-]", "", s)
    if RE_YMD.match(digits):
        return digits

    return None


def list_names(obj_list: Any) -> List[str]:
    """Extract a stable list of normalized names from list[dict{name}] or list[str]."""
    if not isinstance(obj_list, list):
        return []
    out: List[str] = []
    for it in obj_list:
        if isinstance(it, dict):
            name = it.get("name")
            n = norm_text(name)
            if n:
                out.append(n)
        elif isinstance(it, str):
            n = norm_text(it)
            if n:
                out.append(n)
    # unique + stable
    return sorted(set(out))


# -----------------------------
# Fuzzy scoring
# -----------------------------
def fuzzy_ratio(a: Any, b: Any) -> float:
    a_n = norm_text(a)
    b_n = norm_text(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    return _token_set_ratio(a_n, b_n)


def fuzzy_f1_names(pred: List[str], gold: List[str], threshold: float = 0.90) -> float:
    """F1 after greedy bipartite matching above threshold."""
    pred = [norm_text(x) for x in pred if norm_text(x)]
    gold = [norm_text(x) for x in gold if norm_text(x)]

    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    used_g = set()
    tp = 0

    for p in pred:
        best_j = None
        best_s = -1.0
        for j, g in enumerate(gold):
            if j in used_g:
                continue
            s = _token_set_ratio(p, g)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is not None and best_s >= threshold:
            tp += 1
            used_g.add(best_j)

    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


# -----------------------------
# Score dataclass
# -----------------------------
@dataclass
class FieldScores:
    # Fuzzy text
    fuzzy_title: float
    fuzzy_classification: float

    # Exact dates (after normalization)
    exact_pub_date_application: float
    exact_pub_date_publication: float
    exact_pub_date_foreign: float

    # Fuzzy list F1
    f1_inventors: float
    f1_assignees: float

    # Global (simple average)
    global_score: float


def score_one(
    pred: Dict[str, Any],
    gold: Dict[str, Any],
    *,
    name_threshold: float,
) -> FieldScores:
    # predictions may be nested
    if isinstance(pred.get("pred"), dict):
        pred_obj = pred["pred"]
    elif isinstance(pred.get("prediction"), dict):
        pred_obj = pred["prediction"]
    else:
        pred_obj = pred

    # gold may be nested
    gold_obj = gold.get("gold") if isinstance(gold.get("gold"), dict) else gold

    fuzzy_title = fuzzy_ratio(pred_obj.get("title"), gold_obj.get("title"))
    fuzzy_classification = fuzzy_ratio(pred_obj.get("classification"), gold_obj.get("classification"))

    da_pred = norm_date(pred_obj.get("pub_date_application"))
    da_gold = norm_date(gold_obj.get("pub_date_application"))
    exact_pub_date_application = 1.0 if da_pred == da_gold else 0.0

    dp_pred = norm_date(pred_obj.get("pub_date_publication"))
    dp_gold = norm_date(gold_obj.get("pub_date_publication"))
    exact_pub_date_publication = 1.0 if dp_pred == dp_gold else 0.0

    df_pred = norm_date(pred_obj.get("pub_date_foreign"))
    df_gold = norm_date(gold_obj.get("pub_date_foreign"))
    exact_pub_date_foreign = 1.0 if df_pred == df_gold else 0.0

    inv_pred = list_names(pred_obj.get("inventors"))
    inv_gold = list_names(gold_obj.get("inventors"))
    f1_inventors = fuzzy_f1_names(inv_pred, inv_gold, threshold=name_threshold)

    as_pred = list_names(pred_obj.get("assignees"))
    as_gold = list_names(gold_obj.get("assignees"))
    f1_assignees = fuzzy_f1_names(as_pred, as_gold, threshold=name_threshold)

    # Global: equal weights by default (simple + transparent)
    parts = [
        fuzzy_title,
        fuzzy_classification,
        exact_pub_date_application,
        exact_pub_date_publication,
        exact_pub_date_foreign,
        f1_inventors,
        f1_assignees,
    ]
    global_score = float(sum(parts) / len(parts))

    return FieldScores(
        fuzzy_title=float(fuzzy_title),
        fuzzy_classification=float(fuzzy_classification),
        exact_pub_date_application=float(exact_pub_date_application),
        exact_pub_date_publication=float(exact_pub_date_publication),
        exact_pub_date_foreign=float(exact_pub_date_foreign),
        f1_inventors=float(f1_inventors),
        f1_assignees=float(f1_assignees),
        global_score=float(global_score),
    )


def mean_scores(scores: List[FieldScores]) -> Dict[str, float]:
    if not scores:
        return {}
    keys = scores[0].__dict__.keys()
    return {k: float(sum(getattr(s, k) for s in scores) / len(scores)) for k in keys}


def bootstrap_ci(scores: List[FieldScores], n_boot: int = 1000, seed: int = 0) -> Dict[str, Dict[str, float]]:
    rng = random.Random(seed)
    if not scores:
        return {}
    keys = list(scores[0].__dict__.keys())

    boot_vals: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = [scores[rng.randrange(0, len(scores))] for _ in range(len(scores))]
        m = mean_scores(sample)
        for k in keys:
            boot_vals[k].append(m[k])

    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = sorted(boot_vals[k])
        lo = vals[int(0.025 * (len(vals) - 1))]
        hi = vals[int(0.975 * (len(vals) - 1))]
        out[k] = {"lo95": float(lo), "hi95": float(hi)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-jsonl", required=True, type=str)
    ap.add_argument("--pred-jsonl", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)

    # fuzzy thresholds
    ap.add_argument("--name-threshold", type=float, default=0.90, help="Match threshold for inventor/assignee names (0..1)")

    args = ap.parse_args()

    gold_path = Path(args.gold_jsonl)
    pred_path = Path(args.pred_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_jsonl(gold_path)
    preds = load_jsonl(pred_path)

    gold_map = {}
    for x in gold:
        i = get_id(x)
        if i:
            gold_map[i] = x

    pred_map = {}
    for x in preds:
        i = get_id(x)
        if i:
            pred_map[i] = x

    common_ids = sorted(set(gold_map.keys()) & set(pred_map.keys()))
    if not common_ids:
        # helpful debug info
        gold_sample = sorted(list(gold_map.keys()))[:10]
        pred_sample = sorted(list(pred_map.keys()))[:10]
        raise RuntimeError(
            "No common ids between gold and predictions.\n"
            f"Gold sample ids: {gold_sample}\n"
            f"Pred sample ids: {pred_sample}\n"
            "Check identifier/doc_id/file_name fields and extension normalization."
        )

    per_doc_rows = []
    scores: List[FieldScores] = []
    for doc_id in common_ids:
        s = score_one(pred_map[doc_id], gold_map[doc_id], name_threshold=args.name_threshold)
        scores.append(s)
        row = {"identifier": doc_id, **s.__dict__}
        per_doc_rows.append(row)

    df = pd.DataFrame(per_doc_rows).sort_values("identifier")
    (out_dir / "scores_per_doc.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    mean = mean_scores(scores)
    ci = bootstrap_ci(scores, n_boot=args.n_boot, seed=args.seed)

    summary = {
        "n_docs": len(common_ids),
        "settings": {
            "name_threshold": args.name_threshold,
            "bootstrap": {"n_boot": args.n_boot, "seed": args.seed},
            "fuzzy_engine": "rapidfuzz" if "rapidfuzz" in str(_token_set_ratio.__module__) else "difflib",
        },
        "mean": mean,
        "ci95_bootstrap": ci,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Wrote:", out_dir / "scores_per_doc.csv")
    print("✅ Wrote:", out_dir / "summary.json")
    print((out_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
