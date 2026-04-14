from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from .models import PatentMetadata


StrategyName = Literal[
    "baseline",
    "chunked",
    "header_first",
    "two_pass_targeted",
    "self_consistency",
]
MergePolicy = Literal["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"]

STRATEGY_NAMES = (
    "baseline",
    "chunked",
    "header_first",
    "two_pass_targeted",
    "self_consistency",
)
DEFAULT_STRATEGY: StrategyName = "baseline"


def truncate_ocr(extractor: Any, text: str) -> str:
    if len(text) > extractor.max_ocr_chars:
        return text[: extractor.max_ocr_chars] + "\n[...] (truncated)"
    return text


def should_use_chunked(extractor: Any, ocr_text: str) -> bool:
    if extractor.extraction_mode == "chunked":
        return True
    if extractor.extraction_mode == "single":
        return False
    return len(ocr_text) > extractor.max_ocr_chars


def split_text_chunks(extractor: Any, text: str, offset: int = 0) -> List[str]:
    if not text:
        return []

    step = extractor.chunk_size_chars - extractor.chunk_overlap_chars
    start = max(0, int(offset))
    chunks: List[str] = []

    while start < len(text):
        end = min(start + extractor.chunk_size_chars, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


def chunk_offsets(extractor: Any) -> List[int]:
    if extractor.extraction_passes <= 1:
        return [0]
    step = max(1, extractor.chunk_size_chars // extractor.extraction_passes)
    offsets = [0]
    for index in range(1, extractor.extraction_passes):
        offsets.append(index * step)
    return sorted(set(offsets))


def timing_dict(
    extractor: Any,
    *,
    t0: Optional[float],
    t_prompt0: Optional[float],
    t_prompt1: Optional[float],
    t_gen0: Optional[float],
    t_gen1: Optional[float],
    t_parse0: Optional[float],
    t_parse1: Optional[float],
    t_end: Optional[float],
) -> Optional[Dict[str, float]]:
    if extractor.timings == "off" or t0 is None or t_end is None:
        return None

    out: Dict[str, float] = {"t_total_s": max(0.0, t_end - t0)}
    if t_gen0 is not None and t_gen1 is not None:
        out["t_generate_s"] = max(0.0, t_gen1 - t_gen0)

    if extractor.timings == "detailed":
        if t_prompt0 is not None and t_prompt1 is not None:
            out["t_prompt_s"] = max(0.0, t_prompt1 - t_prompt0)
        if t_parse0 is not None and t_parse1 is not None:
            out["t_parse_s"] = max(0.0, t_parse1 - t_parse0)

    return out


def extract_single_metadata(
    extractor: Any,
    ocr_text: str,
    *,
    debug: bool = False,
    truncate: bool = True,
    prompt_suffix_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
    do_sample_override: Optional[bool] = None,
) -> Tuple[PatentMetadata, Optional[Dict[str, float]], Optional[str]]:
    t0 = time.perf_counter() if extractor.timings != "off" else None
    text_for_prompt = truncate_ocr(extractor, ocr_text) if truncate else ocr_text

    t_prompt0 = time.perf_counter() if extractor.timings == "detailed" else None
    suffix = prompt_suffix_override if prompt_suffix_override is not None else extractor.prompt_suffix
    prompt = extractor._render_prompt_template(extractor.prompt_template, text_for_prompt) + suffix
    t_prompt1 = time.perf_counter() if extractor.timings == "detailed" else None

    if debug:
        print("=" * 80)
        print("📝 PROMPT ENVOYÉ AU MODÈLE:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

    t_gen0 = time.perf_counter() if extractor.timings != "off" else None
    raw_output = extractor._generate(
        prompt,
        temperature_override=temperature_override,
        do_sample_override=do_sample_override,
    )
    t_gen1 = time.perf_counter() if extractor.timings != "off" else None

    if debug:
        print("\n" + "=" * 80)
        print("🤖 SORTIE BRUTE DU MODÈLE:")
        print("=" * 80)
        print(raw_output)
        print("=" * 80)

    t_parse0 = time.perf_counter() if extractor.timings == "detailed" else None
    json_str = extractor._extract_json(raw_output)

    if debug:
        print("\n" + "=" * 80)
        print("📦 JSON EXTRAIT:")
        print("=" * 80)
        print(json_str)
        print("=" * 80 + "\n")

    metadata = extractor._parse_and_validate(json_str)
    t_parse1 = time.perf_counter() if extractor.timings == "detailed" else None
    t_end = time.perf_counter() if extractor.timings != "off" else None

    timing = extractor._timing_dict(
        t0=t0,
        t_prompt0=t_prompt0,
        t_prompt1=t_prompt1,
        t_gen0=t_gen0,
        t_gen1=t_gen1,
        t_parse0=t_parse0,
        t_parse1=t_parse1,
        t_end=t_end,
    )
    raw_out = raw_output if extractor.save_raw_output else None
    return metadata, timing, raw_out


def extract_chunked_metadata(
    extractor: Any,
    ocr_text: str,
    *,
    debug: bool = False,
    merge_policy: Optional[MergePolicy] = None,
    prompt_suffix_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
    do_sample_override: Optional[bool] = None,
) -> Tuple[PatentMetadata, Optional[Dict[str, float]], Optional[List[str]]]:
    t0 = time.perf_counter() if extractor.timings != "off" else None

    candidates: List[PatentMetadata] = []
    raw_outputs: List[str] = []
    sum_generate = 0.0
    sum_prompt = 0.0
    sum_parse = 0.0
    chunk_count = 0
    pass_count = 0

    for offset in extractor._chunk_offsets():
        chunks = extractor._split_text_chunks(ocr_text, offset=offset)
        if not chunks:
            continue
        pass_count += 1
        for chunk in chunks:
            chunk_count += 1
            metadata, timing, raw_output = extractor._extract_single_metadata(
                chunk,
                debug=debug,
                truncate=False,
                prompt_suffix_override=prompt_suffix_override,
                temperature_override=temperature_override,
                do_sample_override=do_sample_override,
            )
            candidates.append(metadata)
            if raw_output is not None:
                raw_outputs.append(raw_output)
            if timing:
                sum_generate += float(timing.get("t_generate_s", 0.0))
                sum_prompt += float(timing.get("t_prompt_s", 0.0))
                sum_parse += float(timing.get("t_parse_s", 0.0))

    if not candidates:
        return extractor._extract_single_metadata(ocr_text, debug=debug, truncate=True)

    t_merge0 = time.perf_counter() if extractor.timings != "off" else None
    merged = extractor._merge_metadata_candidates(candidates, policy=merge_policy)
    t_merge1 = time.perf_counter() if extractor.timings != "off" else None

    if extractor.timings == "off":
        return merged, None, (raw_outputs if extractor.save_raw_output else None)

    t_end = time.perf_counter()
    timing: Dict[str, float] = {
        "t_total_s": max(0.0, t_end - (t0 or t_end)),
        "t_generate_s": max(0.0, sum_generate),
        "n_chunks": float(chunk_count),
        "n_passes": float(pass_count),
    }
    if extractor.timings == "detailed":
        timing["t_prompt_s"] = max(0.0, sum_prompt)
        timing["t_parse_s"] = max(0.0, sum_parse)
    if t_merge0 is not None and t_merge1 is not None:
        timing["t_chunk_merge_s"] = max(0.0, t_merge1 - t_merge0)

    return merged, timing, (raw_outputs if extractor.save_raw_output else None)


def run_baseline_strategy(
    extractor: Any,
    ocr_text: str,
    *,
    debug: bool = False,
    prompt_suffix_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
    do_sample_override: Optional[bool] = None,
) -> Tuple[PatentMetadata, Optional[Dict[str, float]], Any, Dict[str, Any]]:
    if extractor._should_use_chunked(ocr_text):
        metadata, timing, raw_output = extractor._extract_chunked_metadata(
            ocr_text,
            debug=debug,
            merge_policy=None,
            prompt_suffix_override=prompt_suffix_override,
            temperature_override=temperature_override,
            do_sample_override=do_sample_override,
        )
        chunks_count = int((timing or {}).get("n_chunks", 0))
        model_calls = chunks_count if chunks_count > 0 else 1
        merge_policy_used = "legacy_baseline"
    else:
        metadata, timing, raw_output = extractor._extract_single_metadata(
            ocr_text,
            debug=debug,
            truncate=True,
            prompt_suffix_override=prompt_suffix_override,
            temperature_override=temperature_override,
            do_sample_override=do_sample_override,
        )
        chunks_count = 1
        model_calls = 1
        merge_policy_used = "n/a_single"

    meta = {
        "was_rerun": False,
        "pass_count": model_calls,
        "chunks_count": chunks_count,
        "header_first_used": False,
        "merge_policy_used": merge_policy_used,
        "self_consistency_n_used": 1,
        "timing": timing or {},
    }
    return metadata, timing, raw_output, meta


def run_chunked_strategy(extractor: Any, ocr_text: str, *, debug: bool = False) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
    metadata, timing, raw_output = extractor._extract_chunked_metadata(
        ocr_text,
        debug=debug,
        merge_policy=extractor.merge_policy,
    )
    chunks_count = int((timing or {}).get("n_chunks", 0))
    model_calls = chunks_count if chunks_count > 0 else 1
    meta = {
        "strategy_used": "chunked",
        "was_rerun": False,
        "pass_count": model_calls,
        "chunks_count": chunks_count,
        "header_first_used": False,
        "merge_policy_used": extractor.merge_policy,
        "self_consistency_n_used": 1,
        "timing": timing or {},
    }
    return metadata, meta, raw_output


def run_header_first_strategy(extractor: Any, ocr_text: str, *, debug: bool = False) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
    lines = ocr_text.splitlines()
    header_text = "\n".join(lines[: extractor.header_lines]) if lines else ocr_text

    header_meta, timing_header, raw_header = extractor._extract_single_metadata(
        header_text,
        debug=debug,
        truncate=False,
    )
    missing = extractor._missing_critical_fields(header_meta)
    fallback_to_full = bool(missing)

    if fallback_to_full:
        full_meta, timing_full, raw_full, base_meta = extractor._run_baseline_strategy(ocr_text, debug=debug)
        merged = extractor._merge_metadata_candidates([header_meta, full_meta], policy=extractor.merge_policy)
        timing: Dict[str, float] = {}
        if timing_header:
            timing.update({f"header_{key}": value for key, value in timing_header.items()})
        if timing_full:
            timing.update({f"pass1_{key}": value for key, value in timing_full.items()})
        raw_output = {"header": raw_header, "full_text": raw_full}
        pass_count = 1 + int(base_meta.get("pass_count", 1))
        chunks_count = 1 + int(base_meta.get("chunks_count", 1))
    else:
        merged = header_meta
        timing = {f"header_{key}": value for key, value in (timing_header or {}).items()}
        raw_output = {"header": raw_header}
        pass_count = 1
        chunks_count = 1

    meta = {
        "strategy_used": "header_first",
        "was_rerun": fallback_to_full,
        "pass_count": pass_count,
        "chunks_count": chunks_count,
        "header_first_used": True,
        "fallback_to_full": fallback_to_full,
        "missing_critical_fields": missing,
        "merge_policy_used": extractor.merge_policy,
        "self_consistency_n_used": 1,
        "timing": timing,
    }
    return merged, meta, raw_output


def run_two_pass_targeted_strategy(extractor: Any, ocr_text: str, *, debug: bool = False) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
    pass1_meta, timing1, raw1, pass1_info = extractor._run_baseline_strategy(ocr_text, debug=debug)
    conf1, conf1_sub = extractor.compute_confidence(pass1_meta, ocr_text)

    was_rerun = conf1 < extractor.targeted_rerun_threshold
    correction_suffix = (
        "\n\nCorrection mode: Re-check missing/uncertain fields and date consistency against the text. "
        "Output only corrected JSON.\n"
    )

    if was_rerun:
        pass2_meta, timing2, raw2, pass2_info = extractor._run_baseline_strategy(
            ocr_text,
            debug=debug,
            prompt_suffix_override=extractor.prompt_suffix + correction_suffix,
        )
        merged = extractor._merge_metadata_candidates([pass1_meta, pass2_meta], policy=extractor.merge_policy)
        timing: Dict[str, float] = {}
        if timing1:
            timing.update({f"pass1_{key}": value for key, value in timing1.items()})
        if timing2:
            timing.update({f"pass2_{key}": value for key, value in timing2.items()})
        pass_count = int(pass1_info.get("pass_count", 1)) + int(pass2_info.get("pass_count", 1))
        chunks_count = int(pass1_info.get("chunks_count", 1)) + int(pass2_info.get("chunks_count", 1))
        raw_output = {"pass1": raw1, "pass2": raw2}
    else:
        merged = pass1_meta
        timing = {f"pass1_{key}": value for key, value in (timing1 or {}).items()}
        pass_count = int(pass1_info.get("pass_count", 1))
        chunks_count = int(pass1_info.get("chunks_count", 1))
        raw_output = {"pass1": raw1}

    meta = {
        "strategy_used": "two_pass_targeted",
        "was_rerun": was_rerun,
        "targeted_rerun_threshold": extractor.targeted_rerun_threshold,
        "pass1_confidence": conf1,
        "pass1_confidence_subscores": conf1_sub,
        "pass_count": pass_count,
        "chunks_count": chunks_count,
        "header_first_used": False,
        "merge_policy_used": extractor.merge_policy,
        "self_consistency_n_used": 1,
        "timing": timing,
    }
    return merged, meta, raw_output


def field_variance(extractor: Any, rows: List[Dict[str, Any]], field: str) -> float:
    values = [row.get(field) for row in rows if not extractor._is_missing(row.get(field))]
    if len(values) <= 1:
        return 0.0
    uniq = len({extractor._normalize_for_vote(value) for value in values})
    return max(0.0, min(1.0, (uniq - 1) / (len(values) - 1)))


def run_self_consistency_strategy(extractor: Any, ocr_text: str, *, debug: bool = False) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
    n = max(1, extractor.self_consistency_n)
    candidates: List[PatentMetadata] = []
    all_timings: List[Dict[str, float]] = []
    all_raw: List[Any] = []
    pass_count = 0
    chunks_count = 0

    t_merge0 = time.perf_counter() if extractor.timings != "off" else None
    for _ in range(n):
        pred, timing, raw_out, info = extractor._run_baseline_strategy(
            ocr_text,
            debug=debug,
            temperature_override=extractor.self_consistency_temp,
            do_sample_override=(extractor.self_consistency_temp > 0.0 or extractor.do_sample),
        )
        candidates.append(pred)
        pass_count += int(info.get("pass_count", 1))
        chunks_count += int(info.get("chunks_count", 1))
        if timing:
            all_timings.append(timing)
        all_raw.append(raw_out)

    merged = extractor._merge_metadata_candidates(
        candidates,
        policy="vote_majority",
        fallback_policy="prefer_non_null",
    )
    t_merge1 = time.perf_counter() if extractor.timings != "off" else None

    agg_timing: Dict[str, float] = {}
    if all_timings:
        for timing in all_timings:
            for key, value in timing.items():
                agg_timing[key] = agg_timing.get(key, 0.0) + float(value)
    if t_merge0 is not None and t_merge1 is not None:
        agg_timing["t_self_consistency_merge_s"] = max(0.0, t_merge1 - t_merge0)

    rows = [candidate.model_dump(mode="json") for candidate in candidates]
    fields = [
        "title",
        "inventors",
        "assignees",
        "pub_date_application",
        "pub_date_publication",
        "pub_date_foreign",
        "classification",
        "industrial_field",
    ]
    variance_by_field = {field: round(extractor._field_variance(rows, field), 6) for field in fields}
    variance_mean = round(sum(variance_by_field.values()) / len(variance_by_field), 6)

    meta = {
        "strategy_used": "self_consistency",
        "was_rerun": n > 1,
        "pass_count": pass_count,
        "chunks_count": chunks_count,
        "header_first_used": False,
        "merge_policy_used": "vote_majority",
        "self_consistency_n_used": n,
        "self_consistency_temp_used": extractor.self_consistency_temp,
        "self_consistency_variance": variance_mean,
        "self_consistency_variance_by_field": variance_by_field,
        "timing": agg_timing,
    }
    return merged, meta, all_raw


def run_strategy(extractor: Any, ocr_text: str, *, debug: bool = False) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
    if extractor.strategy == "baseline":
        pred, timing, raw_output, base_meta = extractor._run_baseline_strategy(ocr_text, debug=debug)
        base_meta["strategy_used"] = "baseline"
        base_meta["timing"] = timing or {}
        return pred, base_meta, raw_output
    if extractor.strategy == "chunked":
        return extractor._run_chunked_strategy(ocr_text, debug=debug)
    if extractor.strategy == "header_first":
        return extractor._run_header_first_strategy(ocr_text, debug=debug)
    if extractor.strategy == "two_pass_targeted":
        return extractor._run_two_pass_targeted_strategy(ocr_text, debug=debug)
    if extractor.strategy == "self_consistency":
        return extractor._run_self_consistency_strategy(ocr_text, debug=debug)
    raise ValueError(f"Unknown strategy: {extractor.strategy}")


__all__ = [
    "DEFAULT_STRATEGY",
    "MergePolicy",
    "STRATEGY_NAMES",
    "StrategyName",
    "chunk_offsets",
    "extract_chunked_metadata",
    "extract_single_metadata",
    "field_variance",
    "run_baseline_strategy",
    "run_chunked_strategy",
    "run_header_first_strategy",
    "run_self_consistency_strategy",
    "run_strategy",
    "run_two_pass_targeted_strategy",
    "should_use_chunked",
    "split_text_chunks",
    "timing_dict",
    "truncate_ocr",
]
