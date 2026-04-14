from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ExtractionConfig, ProfileConfig, StrategyConfig, VLLMConfig


PACKAGE_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
PROFILE_DEFS_DIR = PACKAGE_ROOT / "profile_defs"
DEFAULT_PROFILE_NAME = "de_legacy_v4"


def resolve_profile_path(name: str) -> Path:
    path = PROFILE_DEFS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown profile {name!r}: {path} does not exist")
    return path


def _resolve_prompt_path(
    *,
    profile_path: Path,
    prompt_resource: Optional[str],
    prompt_path: Optional[str],
    extraction_prompt_path: Optional[str],
) -> Optional[Path]:
    if prompt_path:
        return Path(prompt_path).expanduser().resolve()
    if prompt_resource:
        resolved = (PROMPTS_DIR / prompt_resource).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Prompt resource not found: {resolved}")
        return resolved
    if extraction_prompt_path:
        resolved = (profile_path.parent / extraction_prompt_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Prompt path not found: {resolved}")
        return resolved
    return None


def load_profile(
    *,
    name: str = DEFAULT_PROFILE_NAME,
    profile_path: Optional[str | Path] = None,
    prompt_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ProfileConfig:
    definition_path = Path(profile_path).expanduser().resolve() if profile_path else resolve_profile_path(name)
    payload = json.loads(definition_path.read_text(encoding="utf-8"))

    extraction_payload = dict(payload.get("extraction") or {})
    vllm_payload = dict(payload.get("vllm") or {})
    strategy_payload = dict(payload.get("strategy") or {})

    prompt_resource = payload.get("prompt_resource")
    extraction_prompt_path = extraction_payload.pop("prompt_path", None)
    resolved_prompt_path = _resolve_prompt_path(
        profile_path=definition_path,
        prompt_resource=prompt_resource,
        prompt_path=str(prompt_path) if prompt_path is not None else None,
        extraction_prompt_path=extraction_prompt_path,
    )

    extraction = ExtractionConfig(
        model_name=extraction_payload.get("model_name", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"),
        backend="vllm",
        torch_dtype=extraction_payload.get("torch_dtype", "auto"),
        prompt_id=extraction_payload.get("prompt_id"),
        prompt_path=resolved_prompt_path,
        guardrail_profile=extraction_payload.get("guardrail_profile", "auto"),
        max_ocr_chars=int(extraction_payload.get("max_ocr_chars", 10000)),
        max_new_tokens=int(extraction_payload.get("max_new_tokens", 1024)),
        temperature=float(extraction_payload.get("temperature", 0.0)),
        do_sample=bool(extraction_payload.get("do_sample", False)),
        save_strategy_meta=bool(extraction_payload.get("save_strategy_meta", False)),
        save_raw_output=bool(extraction_payload.get("save_raw_output", False)),
        timings=extraction_payload.get("timings", "basic"),
        vllm=VLLMConfig(
            enable_prefix_caching=bool(vllm_payload.get("enable_prefix_caching", False)),
            tensor_parallel_size=int(vllm_payload.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(vllm_payload.get("gpu_memory_utilization", 0.9)),
            max_model_len=vllm_payload.get("max_model_len"),
            swap_space=float(vllm_payload.get("swap_space", 4.0)),
            enforce_eager=bool(vllm_payload.get("enforce_eager", False)),
            doc_batch_size=vllm_payload.get("doc_batch_size", 32),
            sort_by_prompt_length=bool(vllm_payload.get("sort_by_prompt_length", True)),
            tokenizer_mode=vllm_payload.get("tokenizer_mode", "auto"),
            quantization=vllm_payload.get("quantization", "none"),
        ),
        strategy=StrategyConfig(
            name=strategy_payload.get("name", "baseline"),
            extraction_mode=strategy_payload.get("extraction_mode", "auto"),
            chunk_size_chars=int(strategy_payload.get("chunk_size_chars", 7000)),
            chunk_overlap_chars=int(strategy_payload.get("chunk_overlap_chars", 800)),
            extraction_passes=int(strategy_payload.get("extraction_passes", 2)),
            header_lines=int(strategy_payload.get("header_lines", 30)),
            targeted_rerun_threshold=float(strategy_payload.get("targeted_rerun_threshold", 0.6)),
            self_consistency_n=int(strategy_payload.get("self_consistency_n", 2)),
            self_consistency_temp=float(strategy_payload.get("self_consistency_temp", 0.2)),
            merge_policy=strategy_payload.get("merge_policy", "prefer_non_null"),
        ),
    )

    if overrides:
        extraction = extraction.with_overrides(**overrides)

    return ProfileConfig(
        name=payload.get("name", name),
        description=payload.get("description", ""),
        definition_path=definition_path,
        extraction=extraction,
    )


__all__ = ["DEFAULT_PROFILE_NAME", "PROMPTS_DIR", "PROFILE_DEFS_DIR", "load_profile", "resolve_profile_path"]
