from __future__ import annotations

import json
import unicodedata
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from typing import Any, Dict, List, Literal, Optional, Tuple

import regex as re
from pydantic import ValidationError

from .models import PatentMetadata


MergePolicy = Literal["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"]

_REQUIRED_FIELDS = (
    "title",
    "inventors",
    "assignees",
    "pub_date_application",
    "pub_date_publication",
    "pub_date_foreign",
    "classification",
    "industrial_field",
)


def first_non_empty(values: List[Any]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def longest_non_empty(values: List[Any]) -> Optional[str]:
    candidates = [value.strip() for value in values if isinstance(value, str) and value.strip()]
    if not candidates:
        return None
    return max(candidates, key=len)


def to_date(value: Any) -> Optional[date]:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except Exception:
            return None
    return None


def normalize_entity_list(value: Any) -> Optional[List[Dict[str, Optional[str]]]]:
    if value is None:
        return None

    def _norm_entity(name: Any, address: Any) -> Optional[Dict[str, Optional[str]]]:
        name = str(name or "").strip()
        if not name:
            return None
        address = str(address).strip() if address is not None else None
        if not address:
            address = None
        return {"name": name, "address": address}

    def _dedupe_entities(items: List[Dict[str, Optional[str]]]) -> Optional[List[Dict[str, Optional[str]]]]:
        seen = set()
        out: List[Dict[str, Optional[str]]] = []
        for item in items:
            normalized = _norm_entity(item.get("name"), item.get("address"))
            if normalized is None:
                continue
            key = (
                re.sub(r"\s+", " ", normalized["name"].lower()),
                re.sub(r"\s+", " ", (normalized["address"] or "").lower()),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(normalized)
        return out or None

    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return _dedupe_entities(value)

    if isinstance(value, str):
        entities: List[Dict[str, Optional[str]]] = []
        for chunk in value.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            match = re.match(r"(.+?)\s*\(([^)]+)\)", chunk)
            if match:
                entities.append({"name": match.group(1).strip(), "address": match.group(2).strip()})
            else:
                entities.append({"name": chunk, "address": None})
        return _dedupe_entities(entities)

    return None


def normalize_identity_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def same_person_identity(left: Any, right: Any, *, threshold: float = 0.92) -> bool:
    left_norm = normalize_identity_name(left)
    right_norm = normalize_identity_name(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    return SequenceMatcher(None, left_norm, right_norm).ratio() >= threshold


def merge_entity_lists(values: List[Any]) -> Optional[List[Dict[str, Optional[str]]]]:
    seen = set()
    out: List[Dict[str, Optional[str]]] = []

    for value in values:
        normalized = normalize_entity_list(value)
        if not normalized:
            continue
        for item in normalized:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            address = item.get("address")
            address = str(address).strip() if address is not None else None
            if not address:
                address = None
            key = (
                re.sub(r"\s+", " ", name.lower()),
                re.sub(r"\s+", " ", (address or "").lower()),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "address": address})

    return out or None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def normalize_for_vote(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip().lower())
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def choose_scalar(
    values: List[Any],
    policy: MergePolicy,
    fallback_policy: MergePolicy = "prefer_non_null",
) -> Any:
    if not values:
        return None

    if policy == "prefer_first":
        return values[0]
    if policy == "prefer_last":
        return values[-1]
    if policy == "prefer_non_null":
        for value in values:
            if not is_missing(value):
                return value
        return None

    non_missing = [value for value in values if not is_missing(value)]
    if non_missing:
        counts = Counter(normalize_for_vote(value) for value in non_missing)
        winner_key, _ = counts.most_common(1)[0]
        for value in non_missing:
            if normalize_for_vote(value) == winner_key:
                return value
    if fallback_policy != "vote_majority":
        return choose_scalar(values, fallback_policy, fallback_policy="prefer_non_null")
    return None


def enforce_date_order(data: Dict[str, Any]) -> Dict[str, Any]:
    pub_date = to_date(data.get("pub_date_publication"))
    app_date = to_date(data.get("pub_date_application"))
    foreign_date = to_date(data.get("pub_date_foreign"))

    if pub_date and app_date and app_date > pub_date:
        app_date = None
    if app_date and foreign_date and foreign_date > app_date:
        foreign_date = None
    if pub_date and not app_date and foreign_date and foreign_date > pub_date:
        foreign_date = None

    data["pub_date_publication"] = pub_date.isoformat() if pub_date else None
    data["pub_date_application"] = app_date.isoformat() if app_date else None
    data["pub_date_foreign"] = foreign_date.isoformat() if foreign_date else None
    return data


def is_company_name(name: str) -> bool:
    if not name:
        return False

    name_lower = name.lower()
    company_patterns = [
        r"\&",
        r"\bund\b",
        r"\bet\b",
        r"\bgmbh\b",
        r"\bag\b",
        r"\bsa\b",
        r"\bco\.",
        r"\bkg\b",
        r"\bltd\b",
        r"\binc\b",
        r"\bcorp\b",
        r"\bs\.a\.",
        r"\bs\.r\.l\.",
    ]
    for pattern in company_patterns:
        if re.search(pattern, name_lower):
            return True

    letters = [char for char in name if char.isalpha()]
    if letters:
        upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
        if upper_ratio > 0.6:
            return True

    return False


def fix_inventor_assignee_confusion(data: Dict[str, Any]) -> Dict[str, Any]:
    inventors = data.get("inventors") or []
    assignees = data.get("assignees") or []

    if not inventors:
        return data

    true_inventors = []
    misplaced_companies = []

    for inventor in inventors:
        if isinstance(inventor, dict):
            name = inventor.get("name", "")
            if is_company_name(name):
                misplaced_companies.append(inventor)
            else:
                true_inventors.append(inventor)

    if misplaced_companies:
        print(f"🔧 Correction : {len(misplaced_companies)} entreprise(s) déplacée(s) vers assignees")
        data["inventors"] = true_inventors if true_inventors else None
        data["assignees"] = (assignees + misplaced_companies) or None

    return data


def fix_duplicate_dates(data: Dict[str, Any]) -> Dict[str, Any]:
    app_date = data.get("pub_date_application")
    pub_date = data.get("pub_date_publication")
    foreign_date = data.get("pub_date_foreign")

    if app_date and pub_date and app_date == pub_date:
        data["pub_date_application"] = None

    if app_date and pub_date and foreign_date and app_date == pub_date == foreign_date:
        data["pub_date_application"] = None
        data["pub_date_foreign"] = None

    return data


def extract_json(text: str) -> str:
    candidates = [match.group(0) for match in re.finditer(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)]
    if candidates:
        for raw in reversed(candidates):
            try:
                json.loads(raw)
                return raw
            except Exception:
                continue
        return candidates[-1]

    alt = re.search(r'"identifier".*', text, re.DOTALL)
    if alt:
        raw = alt.group(0).strip()
        if not raw.startswith("{"):
            raw = "{\n" + raw
        if not raw.endswith("}"):
            raw += "\n}"
        return raw

    return "{}"


def parse_and_validate(json_str: str) -> PatentMetadata:
    try:
        data = json.loads(json_str)

        if not isinstance(data, dict):
            print(f"⚠️ JSON type inattendu: {type(data)}")
            data = data[0] if isinstance(data, list) and data else {}

        if "assignee" in data and "assignees" not in data:
            data["assignees"] = data.pop("assignee")
        if "inventor" in data and "inventors" not in data:
            data["inventors"] = data.pop("inventor")
        if "class" in data and "classification" not in data:
            data["classification"] = data.pop("class")

        for key in _REQUIRED_FIELDS:
            data.setdefault(key, None)

        data["inventors"] = normalize_entity_list(data.get("inventors"))
        data["assignees"] = normalize_entity_list(data.get("assignees"))
        data = fix_inventor_assignee_confusion(data)
        data = fix_duplicate_dates(data)

        return PatentMetadata(**data)

    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        print(f"⚠️  Erreur de validation JSON: {e}")
        print(f"→ JSON brut:\n{json_str}\n")
        return PatentMetadata()


def merge_metadata_candidates(
    candidates: List[PatentMetadata],
    *,
    policy: Optional[MergePolicy] = None,
    fallback_policy: MergePolicy = "prefer_non_null",
) -> PatentMetadata:
    if not candidates:
        return PatentMetadata()
    if len(candidates) == 1:
        return candidates[0]

    rows = [candidate.model_dump(mode="json") for candidate in candidates]

    if policy is None:
        pub_dates = sorted(date_value for date_value in (to_date(row.get("pub_date_publication")) for row in rows) if date_value)
        app_dates = sorted(date_value for date_value in (to_date(row.get("pub_date_application")) for row in rows) if date_value)
        foreign_dates = sorted(date_value for date_value in (to_date(row.get("pub_date_foreign")) for row in rows) if date_value)

        pub_date = pub_dates[0] if pub_dates else None
        if app_dates:
            app_valid = [date_value for date_value in app_dates if pub_date is None or date_value <= pub_date]
            app_date = app_valid[0] if app_valid else app_dates[0]
        else:
            app_date = None

        if foreign_dates:
            foreign_valid = [date_value for date_value in foreign_dates if app_date is None or date_value <= app_date]
            foreign_date = foreign_valid[0] if foreign_valid else foreign_dates[0]
        else:
            foreign_date = None

        merged_legacy: Dict[str, Any] = {
            "title": longest_non_empty([row.get("title") for row in rows]),
            "inventors": merge_entity_lists([row.get("inventors") for row in rows]),
            "assignees": merge_entity_lists([row.get("assignees") for row in rows]),
            "pub_date_application": app_date.isoformat() if app_date else None,
            "pub_date_publication": pub_date.isoformat() if pub_date else None,
            "pub_date_foreign": foreign_date.isoformat() if foreign_date else None,
            "classification": first_non_empty([row.get("classification") for row in rows]),
            "industrial_field": first_non_empty([row.get("industrial_field") for row in rows]),
        }
        return parse_and_validate(json.dumps(merged_legacy, ensure_ascii=False))

    merged: Dict[str, Any] = {
        "title": choose_scalar([row.get("title") for row in rows], policy, fallback_policy),
        "classification": choose_scalar([row.get("classification") for row in rows], policy, fallback_policy),
        "industrial_field": choose_scalar([row.get("industrial_field") for row in rows], policy, fallback_policy),
        "pub_date_application": choose_scalar([row.get("pub_date_application") for row in rows], policy, fallback_policy),
        "pub_date_publication": choose_scalar([row.get("pub_date_publication") for row in rows], policy, fallback_policy),
        "pub_date_foreign": choose_scalar([row.get("pub_date_foreign") for row in rows], policy, fallback_policy),
        "inventors": merge_entity_lists([row.get("inventors") for row in rows]),
        "assignees": merge_entity_lists([row.get("assignees") for row in rows]),
    }
    merged = enforce_date_order(merged)
    return parse_and_validate(json.dumps(merged, ensure_ascii=False))


def missing_critical_fields(metadata: PatentMetadata) -> List[str]:
    data = metadata.model_dump(mode="json")
    missing: List[str] = []
    for key in ("title", "inventors", "assignees"):
        if is_missing(data.get(key)):
            missing.append(key)
    if is_missing(data.get("pub_date_application")):
        missing.append("pub_date_application")
    if is_missing(data.get("pub_date_publication")):
        missing.append("pub_date_publication")
    return missing


def date_coherence_subscore(data: Dict[str, Any]) -> float:
    app_date = to_date(data.get("pub_date_application"))
    pub_date = to_date(data.get("pub_date_publication"))
    foreign_date = to_date(data.get("pub_date_foreign"))
    checks = 0
    valid = 0
    if app_date and pub_date:
        checks += 1
        valid += int(app_date <= pub_date)
    if foreign_date and app_date:
        checks += 1
        valid += int(foreign_date <= app_date)
    if checks == 0:
        return 0.6
    return valid / checks


def entity_subscore(entities: Any) -> float:
    normalized = normalize_entity_list(entities)
    if not normalized:
        return 0.0
    quality = 0.0
    for item in normalized:
        name_ok = bool(str(item.get("name") or "").strip())
        addr_ok = bool(str(item.get("address") or "").strip())
        quality += 0.7 if name_ok else 0.0
        quality += 0.3 if addr_ok else 0.0
    return min(1.0, quality / max(1, len(normalized)))


def compute_confidence(prediction: PatentMetadata, raw_text: str) -> Tuple[float, Dict[str, float]]:
    data = prediction.model_dump(mode="json")
    fields = list(_REQUIRED_FIELDS)
    non_null_count = sum(0 if is_missing(data.get(field)) else 1 for field in fields)
    completeness = non_null_count / len(fields)

    date_coherence = date_coherence_subscore(data)
    inventors_score = entity_subscore(data.get("inventors"))
    assignees_score = entity_subscore(data.get("assignees"))
    entity_quality = (inventors_score + assignees_score) / 2.0

    title = str(data.get("title") or "").strip()
    title_in_text = 1.0 if (title and title in raw_text) else 0.0

    score = (0.45 * completeness) + (0.25 * date_coherence) + (0.20 * entity_quality) + (0.10 * title_in_text)
    score = max(0.0, min(1.0, score))
    subscores = {
        "completeness": round(completeness, 6),
        "date_coherence": round(date_coherence, 6),
        "entity_quality": round(entity_quality, 6),
        "title_in_text": round(title_in_text, 6),
    }
    return round(score, 6), subscores
