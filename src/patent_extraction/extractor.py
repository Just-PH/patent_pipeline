from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from . import guardrails as guardrails_mod
from . import postprocess as postprocess_mod
from . import runtime as runtime_mod
from . import strategies as strategy_mod
from .config import ExtractionConfig, ProfileConfig
from .models import PatentExtraction, PatentMetadata
from .profiles import DEFAULT_PROFILE_NAME, load_profile
from .prompt_templates import JSON_ONLY_SUFFIX, PROMPT_BY_ID, PROMPT_EXTRACTION_V4


_PROMPT_BY_ID: Dict[str, str] = dict(PROMPT_BY_ID)
_JSON_ONLY_SUFFIX = JSON_ONLY_SUFFIX


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _non_negative_float(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, float(value))


def _vllm_request_timing(output: Any) -> Dict[str, float]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return {}

    arrival = _non_negative_float(getattr(metrics, "arrival_time", None))
    first_scheduled = _non_negative_float(getattr(metrics, "first_scheduled_time", None))
    first_token = _non_negative_float(getattr(metrics, "first_token_time", None))
    finished = _non_negative_float(getattr(metrics, "finished_time", None))

    timing: Dict[str, float] = {}
    if arrival is not None and finished is not None and finished >= arrival:
        timing["t_vllm_request_s"] = finished - arrival
    if arrival is not None and first_token is not None and first_token >= arrival:
        timing["t_vllm_ttft_s"] = first_token - arrival
    if first_token is not None and finished is not None and finished >= first_token:
        timing["t_vllm_decode_s"] = finished - first_token
    if arrival is not None and first_scheduled is not None and first_scheduled >= arrival:
        timing.setdefault("t_vllm_queue_s", first_scheduled - arrival)

    for key, attr in (
        ("t_vllm_queue_s", "time_in_queue"),
        ("t_vllm_scheduler_s", "scheduler_time"),
        ("t_vllm_model_forward_s", "model_forward_time"),
        ("t_vllm_model_execute_s", "model_execute_time"),
    ):
        val = _non_negative_float(getattr(metrics, attr, None))
        if val is not None:
            timing[key] = val

    return timing


def _iter_batches(rows: List[Dict[str, Any]], batch_size: Optional[int]) -> List[List[Dict[str, Any]]]:
    if not rows:
        return []
    if batch_size is None or batch_size <= 0 or batch_size >= len(rows):
        return [rows]
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    preds_path: Path
    run_meta_path: Path
    raw_outputs_dir: Optional[Path]
    docs_written: int
    wall_s: float


class PatentExtractor:
    _first_non_empty = staticmethod(postprocess_mod.first_non_empty)
    _longest_non_empty = staticmethod(postprocess_mod.longest_non_empty)
    _to_date = staticmethod(postprocess_mod.to_date)
    _merge_entity_lists = staticmethod(postprocess_mod.merge_entity_lists)
    _is_missing = staticmethod(postprocess_mod.is_missing)
    _normalize_for_vote = staticmethod(postprocess_mod.normalize_for_vote)
    _choose_scalar = staticmethod(postprocess_mod.choose_scalar)
    _enforce_date_order = staticmethod(postprocess_mod.enforce_date_order)
    _merge_metadata_candidates = staticmethod(postprocess_mod.merge_metadata_candidates)
    _missing_critical_fields = staticmethod(postprocess_mod.missing_critical_fields)
    _date_coherence_subscore = staticmethod(postprocess_mod.date_coherence_subscore)
    _entity_subscore = staticmethod(postprocess_mod.entity_subscore)
    compute_confidence = staticmethod(postprocess_mod.compute_confidence)
    _extract_json = staticmethod(postprocess_mod.extract_json)
    _normalize_entity_list = staticmethod(postprocess_mod.normalize_entity_list)
    _fix_inventor_assignee_confusion = staticmethod(postprocess_mod.fix_inventor_assignee_confusion)
    _fix_duplicate_dates = staticmethod(postprocess_mod.fix_duplicate_dates)
    _parse_and_validate = staticmethod(postprocess_mod.parse_and_validate)

    _truncate_ocr = strategy_mod.truncate_ocr
    _should_use_chunked = strategy_mod.should_use_chunked
    _split_text_chunks = strategy_mod.split_text_chunks
    _chunk_offsets = strategy_mod.chunk_offsets
    _extract_single_metadata = strategy_mod.extract_single_metadata
    _extract_chunked_metadata = strategy_mod.extract_chunked_metadata
    _run_baseline_strategy = strategy_mod.run_baseline_strategy
    _run_chunked_strategy = strategy_mod.run_chunked_strategy
    _run_header_first_strategy = strategy_mod.run_header_first_strategy
    _run_two_pass_targeted_strategy = strategy_mod.run_two_pass_targeted_strategy
    _field_variance = strategy_mod.field_variance
    _run_self_consistency_strategy = strategy_mod.run_self_consistency_strategy
    _run_strategy = strategy_mod.run_strategy
    _timing_dict = strategy_mod.timing_dict

    _generate = runtime_mod.generate_text
    _build_vllm_sampling_params = staticmethod(runtime_mod.build_sampling_params)

    def __init__(
        self,
        *,
        model_name: str,
        torch_dtype: str = "auto",
        quantization: str = "none",
        prompt_id: Optional[str] = None,
        prompt_template: Optional[str] = None,
        guardrail_profile: str = "auto",
        max_ocr_chars: int = 10000,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        extraction_mode: str = "auto",
        chunk_size_chars: int = 7000,
        chunk_overlap_chars: int = 800,
        extraction_passes: int = 2,
        strategy: str = "baseline",
        header_lines: int = 30,
        targeted_rerun_threshold: float = 0.6,
        self_consistency_n: int = 2,
        self_consistency_temp: float = 0.2,
        merge_policy: str = "prefer_non_null",
        save_strategy_meta: bool = False,
        save_raw_output: bool = False,
        timings: str = "off",
        enable_prefix_caching: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        swap_space: float = 4.0,
        enforce_eager: bool = False,
        doc_batch_size: Optional[int] = 32,
        sort_by_prompt_length: bool = True,
        tokenizer_mode: str = "auto",
    ):
        self.model_name = model_name
        self.backend = "vllm"
        self.torch_dtype = str(torch_dtype).strip().lower()
        self.quantization = str(quantization or "none").strip().lower()
        self.max_ocr_chars = int(max_ocr_chars)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample)
        self.extraction_mode = str(extraction_mode)
        self.chunk_size_chars = int(chunk_size_chars)
        self.chunk_overlap_chars = int(chunk_overlap_chars)
        self.extraction_passes = int(extraction_passes)
        self.strategy = str(strategy)
        self.header_lines = int(header_lines)
        self.targeted_rerun_threshold = float(targeted_rerun_threshold)
        self.self_consistency_n = int(self_consistency_n)
        self.self_consistency_temp = float(self_consistency_temp)
        self.merge_policy = str(merge_policy)
        self.save_strategy_meta = bool(save_strategy_meta)
        self.save_raw_output = bool(save_raw_output)
        self.timings = str(timings)
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.max_model_len = None if max_model_len is None else int(max_model_len)
        self.swap_space = float(swap_space)
        self.enforce_eager = bool(enforce_eager)
        self.doc_batch_size = None if doc_batch_size is None else int(doc_batch_size)
        self.sort_by_prompt_length = bool(sort_by_prompt_length)
        self.tokenizer_mode = str(tokenizer_mode).strip().lower()
        self.guardrail_profile = str(guardrail_profile).strip().lower()

        if self.torch_dtype not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("torch_dtype must be one of: auto|bf16|fp16|fp32")
        if self.guardrail_profile not in guardrails_mod.GUARDRAIL_PROFILES:
            raise ValueError(f"guardrail_profile must be one of: {sorted(guardrails_mod.GUARDRAIL_PROFILES)}")
        if self.chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be > 0")
        if self.chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars must be >= 0")
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be < chunk_size_chars")
        if self.extraction_passes < 1:
            raise ValueError("extraction_passes must be >= 1")
        if self.header_lines < 1:
            raise ValueError("header_lines must be >= 1")
        if not (0.0 <= self.targeted_rerun_threshold <= 1.0):
            raise ValueError("targeted_rerun_threshold must be in [0, 1]")
        if self.self_consistency_n < 1:
            raise ValueError("self_consistency_n must be >= 1")
        if self.self_consistency_temp < 0.0:
            raise ValueError("self_consistency_temp must be >= 0")
        if self.timings not in {"off", "basic", "detailed"}:
            raise ValueError("timings must be one of: off|basic|detailed")
        if self.tokenizer_mode not in {"auto", "mistral"}:
            raise ValueError("tokenizer_mode must be one of: auto|mistral")

        self.prompt_suffix = _JSON_ONLY_SUFFIX
        self.prompt_id = prompt_id
        if self.prompt_id is not None:
            if self.prompt_id not in _PROMPT_BY_ID:
                raise ValueError(f"Unknown prompt_id={self.prompt_id!r}. Allowed: {sorted(_PROMPT_BY_ID.keys())}")
            self.prompt_template = _PROMPT_BY_ID[self.prompt_id]
            self.prompt_template_source = f"prompt_id:{self.prompt_id}"
        else:
            self.prompt_template = prompt_template or PROMPT_EXTRACTION_V4
            self.prompt_template_source = "inline_template" if prompt_template else "default:v4"
        if "{text}" not in self.prompt_template:
            raise ValueError("Le template doit contenir le placeholder {text}")
        self.prompt_hash = _sha256(self.prompt_template + self.prompt_suffix)

        self._last_timing: Optional[Dict[str, float]] = None
        self._last_raw_output: Optional[Any] = None
        self._last_strategy_meta: Optional[Dict[str, Any]] = None

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        print("🧠 Backend: vllm")
        print(f"📦 Model: {self.model_name}")
        runtime_mod.load_model(self)

    @staticmethod
    def _render_prompt_template(template: str, text: str) -> str:
        if "{text}" not in template:
            raise ValueError("Le template doit contenir le placeholder {text}")
        sentinel = "__PATENT_EXTRACTION_TEXT_PLACEHOLDER__"
        rendered = template.replace("{text}", sentinel)
        rendered = rendered.replace("{{", "{").replace("}}", "}")
        return rendered.replace(sentinel, text)

    def set_prompt_template(self, template: str) -> None:
        if "{text}" not in template:
            raise ValueError("Le template doit contenir le placeholder {text}")
        self.prompt_id = None
        self.prompt_template = template
        self.prompt_template_source = "inline_template"
        self.prompt_hash = _sha256(self.prompt_template + self.prompt_suffix)

    def _apply_de_legacy_self_applicant_guardrail(self, metadata: PatentMetadata, ocr_text: str) -> PatentMetadata:
        return guardrails_mod.apply_de_legacy_self_applicant_guardrail(
            metadata,
            ocr_text,
            prompt_id=self.prompt_id,
            guardrail_profile=self.guardrail_profile,
        )

    def extract(self, ocr_text: str, debug: bool = False) -> PatentExtraction:
        try:
            metadata, strategy_meta, raw_output = self._run_strategy(ocr_text, debug=debug)
        except Exception as exc:
            print(f"⚠️ Strategy '{self.strategy}' failed ({exc.__class__.__name__}). Fallback to baseline.")
            metadata, timing, raw_output, base_meta = self._run_baseline_strategy(ocr_text, debug=debug)
            strategy_meta = {
                **base_meta,
                "strategy_used": "baseline",
                "strategy_fallback_from": self.strategy,
                "timing": timing or {},
            }

        metadata = self._apply_de_legacy_self_applicant_guardrail(metadata, ocr_text)
        confidence, subscores = self.compute_confidence(metadata, ocr_text)
        strategy_meta["confidence_score"] = confidence
        strategy_meta["confidence_subscores"] = subscores

        self._last_timing = strategy_meta.get("timing") or None
        self._last_raw_output = raw_output
        self._last_strategy_meta = strategy_meta
        return PatentExtraction(ocr_text=ocr_text, model=self.model_name, prediction=metadata)

    def extract_from_file(self, txt_path: Path, raw_output_dir: Optional[Path] = None) -> dict:
        t_file0 = time.perf_counter() if self.timings != "off" else None
        try:
            t_read0 = time.perf_counter() if self.timings == "detailed" else None
            ocr_text = txt_path.read_text(encoding="utf-8", errors="ignore")
            t_read1 = time.perf_counter() if self.timings == "detailed" else None

            if not ocr_text.strip():
                return self._make_empty_ocr_record(txt_path, t_file0=t_file0, t_read0=t_read0, t_read1=t_read1)

            extraction = self.extract(ocr_text)
            record = extraction.model_dump(mode="json")
            record["file_name"] = txt_path.name
            record["ocr_path"] = str(txt_path)
            if isinstance(record.get("prediction"), dict):
                record["prediction"]["identifier"] = txt_path.stem.split("_")[0]
            if self.prompt_id is not None:
                record["prompt_id"] = self.prompt_id
            record["prompt_hash"] = self.prompt_hash

            strategy_meta = self._last_strategy_meta or {}
            record["strategy_used"] = strategy_meta.get("strategy_used", "baseline")
            record["confidence_score"] = strategy_meta.get("confidence_score", 0.0)
            if self.save_strategy_meta:
                record["confidence_subscores"] = strategy_meta.get("confidence_subscores", {})
                record["pass_count"] = strategy_meta.get("pass_count", 1)
                record["was_rerun"] = strategy_meta.get("was_rerun", False)
                record["chunks_count"] = strategy_meta.get("chunks_count", 1)
                record["header_first_used"] = strategy_meta.get("header_first_used", False)
                record["merge_policy_used"] = strategy_meta.get("merge_policy_used", self.merge_policy)
                record["self_consistency_n_used"] = strategy_meta.get("self_consistency_n_used", 1)
                if "fallback_to_full" in strategy_meta:
                    record["fallback_to_full"] = strategy_meta.get("fallback_to_full")
                if "self_consistency_variance" in strategy_meta:
                    record["self_consistency_variance"] = strategy_meta.get("self_consistency_variance")
                if "vllm_doc_batch_size_used" in strategy_meta:
                    record["vllm_doc_batch_size_used"] = strategy_meta.get("vllm_doc_batch_size_used")

            if self.timings != "off":
                timing_out: Dict[str, float] = {}
                if self.timings == "detailed" and t_read0 is not None and t_read1 is not None:
                    timing_out["t_read_s"] = max(0.0, t_read1 - t_read0)
                if self._last_timing:
                    timing_out.update(self._last_timing)
                if t_file0 is not None:
                    timing_out["t_total_file_s"] = max(0.0, time.perf_counter() - t_file0)
                record["timing"] = timing_out

            if self.save_raw_output:
                record["raw_output"] = self._last_raw_output
                if raw_output_dir is not None:
                    raw_output_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = raw_output_dir / f"{txt_path.name}.raw.txt"
                    raw_data = self._last_raw_output
                    if isinstance(raw_data, list):
                        sep = "\n\n" + ("=" * 40) + " CHUNK " + ("=" * 40) + "\n\n"
                        raw_text = sep.join(raw_data)
                    elif isinstance(raw_data, dict):
                        raw_text = json.dumps(raw_data, ensure_ascii=False, indent=2)
                    else:
                        raw_text = raw_data or ""
                    raw_path.write_text(raw_text, encoding="utf-8")
                    record["raw_output_path"] = str(raw_path)
                    record.pop("raw_output", None)

            return record
        except Exception as exc:
            print(f"⚠️ Erreur sur {txt_path.name}: {exc}")
            traceback.print_exc()
            return self._make_exception_record(
                txt_path,
                t_file0=t_file0,
                error_type=exc.__class__.__name__,
                error_detail=str(exc),
            )

    def batch_extract(self, txt_dir: Path, out_file: Path, limit: Optional[int] = None, raw_output_dir: Optional[Path] = None) -> int:
        return self._batch_extract_vllm(txt_dir=txt_dir, out_file=out_file, limit=limit, raw_output_dir=raw_output_dir)

    def _make_empty_ocr_record(
        self,
        txt_path: Path,
        *,
        t_file0: Optional[float],
        t_read0: Optional[float],
        t_read1: Optional[float],
    ) -> Dict[str, Any]:
        rec = {
            "file_name": txt_path.name,
            "ocr_path": str(txt_path),
            "error": "empty_ocr",
            "strategy_used": self.strategy,
            "confidence_score": 0.0,
            "prompt_hash": self.prompt_hash,
        }
        if self.prompt_id is not None:
            rec["prompt_id"] = self.prompt_id
        if self.timings != "off" and t_file0 is not None:
            timing = {"t_total_file_s": max(0.0, time.perf_counter() - t_file0)}
            if self.timings == "detailed" and t_read0 is not None and t_read1 is not None:
                timing["t_read_s"] = max(0.0, t_read1 - t_read0)
            rec["timing"] = timing
        return rec

    def _make_exception_record(
        self,
        txt_path: Path,
        *,
        t_file0: Optional[float],
        error_type: str,
        error_detail: str,
        timing: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        rec = {
            "file_name": txt_path.name,
            "ocr_path": str(txt_path),
            "error": f"exception: {error_type}",
            "error_type": error_type,
            "error_detail": error_detail,
            "strategy_used": self.strategy,
            "confidence_score": 0.0,
            "prompt_hash": self.prompt_hash,
        }
        if self.prompt_id is not None:
            rec["prompt_id"] = self.prompt_id
        if timing:
            rec["timing"] = timing
        elif self.timings != "off" and t_file0 is not None:
            rec["timing"] = {"t_total_file_s": max(0.0, time.perf_counter() - t_file0)}
        return rec

    def _build_success_record(
        self,
        doc_row: Dict[str, Any],
        *,
        metadata: PatentMetadata,
        strategy_meta: Dict[str, Any],
        raw_output: Any,
        raw_output_dir: Optional[Path],
    ) -> Dict[str, Any]:
        txt_path = doc_row["txt_path"]
        ocr_text = doc_row["ocr_text"]
        record = {
            "ocr_text": ocr_text,
            "model": self.model_name,
            "prediction": metadata.model_dump(mode="json"),
            "file_name": txt_path.name,
            "ocr_path": str(txt_path),
            "prompt_hash": self.prompt_hash,
            "strategy_used": strategy_meta.get("strategy_used", self.strategy),
            "confidence_score": strategy_meta.get("confidence_score", 0.0),
        }
        if isinstance(record.get("prediction"), dict):
            record["prediction"]["identifier"] = txt_path.stem.split("_")[0]
        if self.prompt_id is not None:
            record["prompt_id"] = self.prompt_id

        if self.save_strategy_meta:
            record["confidence_subscores"] = strategy_meta.get("confidence_subscores", {})
            record["pass_count"] = strategy_meta.get("pass_count", 1)
            record["was_rerun"] = strategy_meta.get("was_rerun", False)
            record["chunks_count"] = strategy_meta.get("chunks_count", 1)
            record["header_first_used"] = strategy_meta.get("header_first_used", False)
            record["merge_policy_used"] = strategy_meta.get("merge_policy_used", self.merge_policy)
            record["self_consistency_n_used"] = strategy_meta.get("self_consistency_n_used", 1)
            if "fallback_to_full" in strategy_meta:
                record["fallback_to_full"] = strategy_meta.get("fallback_to_full")
            if "self_consistency_variance" in strategy_meta:
                record["self_consistency_variance"] = strategy_meta.get("self_consistency_variance")
            if "vllm_doc_batch_size_used" in strategy_meta:
                record["vllm_doc_batch_size_used"] = strategy_meta.get("vllm_doc_batch_size_used")

        if self.timings != "off":
            timing = strategy_meta.get("timing") or {}
            if timing:
                record["timing"] = timing

        if self.save_raw_output:
            record["raw_output"] = raw_output
            if raw_output_dir is not None:
                raw_output_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_output_dir / f"{txt_path.name}.raw.txt"
                raw_data = raw_output
                if isinstance(raw_data, list):
                    sep = "\n\n" + ("=" * 40) + " CHUNK " + ("=" * 40) + "\n\n"
                    raw_text = sep.join(
                        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, indent=2)
                        for item in raw_data
                    )
                elif isinstance(raw_data, dict):
                    raw_text = json.dumps(raw_data, ensure_ascii=False, indent=2)
                else:
                    raw_text = raw_data or ""
                raw_path.write_text(raw_text, encoding="utf-8")
                record["raw_output_path"] = str(raw_path)
                record.pop("raw_output", None)

        return record

    def _load_vllm_doc_rows(self, txt_files: List[Path]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        doc_rows: List[Dict[str, Any]] = []
        records_by_index: Dict[int, Dict[str, Any]] = {}

        for original_index, txt_path in enumerate(tqdm(txt_files, desc="📄 Chargement OCR vLLM", unit="doc")):
            t_file0 = time.perf_counter() if self.timings != "off" else None
            t_read0 = time.perf_counter() if self.timings == "detailed" else None
            ocr_text = txt_path.read_text(encoding="utf-8", errors="ignore")
            t_read1 = time.perf_counter() if self.timings == "detailed" else None

            if not ocr_text.strip():
                records_by_index[original_index] = self._make_empty_ocr_record(
                    txt_path,
                    t_file0=t_file0,
                    t_read0=t_read0,
                    t_read1=t_read1,
                )
                continue

            doc_rows.append(
                {
                    "original_index": original_index,
                    "txt_path": txt_path,
                    "ocr_text": ocr_text,
                    "t_file0": t_file0,
                    "t_read0": t_read0,
                    "t_read1": t_read1,
                }
            )

        return doc_rows, records_by_index

    def _build_vllm_prompt_row(
        self,
        doc_row: Dict[str, Any],
        *,
        prompt_text: str,
        truncate: bool,
        prompt_suffix_override: Optional[str],
        candidate_order: int,
    ) -> Dict[str, Any]:
        t_prompt0 = time.perf_counter() if self.timings == "detailed" else None
        text_for_prompt = self._truncate_ocr(prompt_text) if truncate else prompt_text
        suffix = prompt_suffix_override if prompt_suffix_override is not None else self.prompt_suffix
        prompt = self._render_prompt_template(self.prompt_template, text_for_prompt) + suffix
        t_prompt1 = time.perf_counter() if self.timings == "detailed" else None
        return {
            **doc_row,
            "prompt": prompt,
            "candidate_order": candidate_order,
            "t_prompt0": t_prompt0,
            "t_prompt1": t_prompt1,
        }

    def _prepare_vllm_prompt_rows(
        self,
        doc_rows: List[Dict[str, Any]],
        *,
        prep_desc: str,
        force_chunked: bool,
        merge_policy: Optional[str],
        prompt_suffix_override: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        prompt_rows: List[Dict[str, Any]] = []
        doc_plans: Dict[int, Dict[str, Any]] = {}

        for doc_row in tqdm(doc_rows, desc=prep_desc, unit="doc"):
            original_index = int(doc_row["original_index"])
            ocr_text = str(doc_row["ocr_text"])
            if force_chunked:
                use_chunked = True
            elif self.extraction_mode == "chunked":
                use_chunked = True
            elif self.extraction_mode == "single":
                use_chunked = False
            else:
                use_chunked = len(ocr_text) > self.max_ocr_chars

            candidate_order = 0
            chunk_count = 0
            if use_chunked:
                for offset in self._chunk_offsets():
                    for chunk in self._split_text_chunks(ocr_text, offset=offset):
                        prompt_rows.append(
                            self._build_vllm_prompt_row(
                                doc_row,
                                prompt_text=chunk,
                                truncate=False,
                                prompt_suffix_override=prompt_suffix_override,
                                candidate_order=candidate_order,
                            )
                        )
                        candidate_order += 1
                        chunk_count += 1

            if not use_chunked or chunk_count == 0:
                prompt_rows.append(
                    self._build_vllm_prompt_row(
                        doc_row,
                        prompt_text=ocr_text,
                        truncate=True,
                        prompt_suffix_override=prompt_suffix_override,
                        candidate_order=candidate_order,
                    )
                )
                use_chunked = False
                chunk_count = 1

            doc_plans[original_index] = {
                **doc_row,
                "use_chunked": use_chunked,
                "pass_count": chunk_count,
                "chunks_count": chunk_count,
                "merge_policy_used": (merge_policy if merge_policy is not None else "legacy_baseline") if use_chunked else "n/a_single",
            }

        return prompt_rows, doc_plans

    def _run_vllm_prompt_batches(
        self,
        prompt_rows: List[Dict[str, Any]],
        *,
        infer_desc: str,
        temperature_override: Optional[float] = None,
        do_sample_override: Optional[bool] = None,
    ) -> Tuple[Dict[int, List[Dict[str, Any]]], float]:
        if not prompt_rows:
            return {}, 0.0

        prompt_results_by_doc: Dict[int, List[Dict[str, Any]]] = {}
        work_rows = list(prompt_rows)
        if self.sort_by_prompt_length:
            work_rows.sort(key=lambda row: len(row["prompt"]), reverse=True)

        sampling_params = self._build_vllm_sampling_params(
            self,
            temperature_override=temperature_override,
            do_sample_override=do_sample_override,
        )

        infer_bar = tqdm(total=len(work_rows), desc=infer_desc, unit="doc")
        batch_generate_total_t = 0.0
        microbatch_id = 0

        try:
            for batch_rows in _iter_batches(work_rows, self.doc_batch_size):
                prompts = [row["prompt"] for row in batch_rows]
                outputs: List[Any] = []
                microbatch_generate_t = 0.0
                t_gen_started_at: Optional[float] = None
                if prompts:
                    t_gen0 = time.perf_counter() if self.timings != "off" else None
                    t_gen_started_at = t_gen0
                    outputs = list(self.model.generate(prompts, sampling_params=sampling_params, use_tqdm=False))
                    if t_gen0 is not None:
                        microbatch_generate_t = max(0.0, time.perf_counter() - t_gen0)
                        batch_generate_total_t += microbatch_generate_t

                current_microbatch_id = microbatch_id
                microbatch_id += 1

                for index, row in enumerate(batch_rows):
                    raw_output = ""
                    output = outputs[index] if index < len(outputs) else None
                    t_parse0 = time.perf_counter() if self.timings == "detailed" else None
                    t_parse1: Optional[float] = None
                    prompt_result: Dict[str, Any]
                    try:
                        if output is None:
                            raise RuntimeError("vLLM returned fewer outputs than prompts")
                        candidates = getattr(output, "outputs", None) or []
                        raw_output = (getattr(candidates[0], "text", None) or "") if candidates else ""
                        json_str = self._extract_json(raw_output)
                        metadata = self._parse_and_validate(json_str)
                        metadata = self._apply_de_legacy_self_applicant_guardrail(metadata, str(row.get("ocr_text") or ""))
                        prompt_result = {"metadata": metadata, "raw_output": raw_output}
                    except Exception as exc:
                        prompt_result = {
                            "error": f"exception: {exc.__class__.__name__}",
                            "error_type": exc.__class__.__name__,
                            "error_detail": str(exc),
                            "raw_output": raw_output,
                        }

                    if self.timings == "detailed":
                        t_parse1 = time.perf_counter()

                    if self.timings != "off":
                        timing_out: Dict[str, float] = {"t_batch_generate_s": microbatch_generate_t}
                        if output is not None:
                            timing_out.update(_vllm_request_timing(output))

                        read_s = 0.0
                        prompt_s = 0.0
                        parse_s = 0.0
                        prebatch_wait_s = 0.0
                        if self.timings == "detailed":
                            t_read0 = row.get("t_read0")
                            t_read1 = row.get("t_read1")
                            t_prompt0 = row.get("t_prompt0")
                            t_prompt1 = row.get("t_prompt1")
                            if t_read0 is not None and t_read1 is not None:
                                read_s = max(0.0, t_read1 - t_read0)
                                timing_out["t_read_s"] = read_s
                            if t_prompt0 is not None and t_prompt1 is not None:
                                prompt_s = max(0.0, t_prompt1 - t_prompt0)
                                timing_out["t_prompt_s"] = prompt_s
                            if t_parse0 is not None and t_parse1 is not None:
                                parse_s = max(0.0, t_parse1 - t_parse0)
                                timing_out["t_parse_s"] = parse_s
                            if t_gen_started_at is not None and t_prompt1 is not None:
                                prebatch_wait_s = max(0.0, t_gen_started_at - t_prompt1)
                                timing_out["t_prebatch_wait_s"] = prebatch_wait_s

                        request_s = timing_out.get("t_vllm_request_s")
                        if request_s is not None:
                            timing_out["t_total_file_s"] = read_s + prompt_s + prebatch_wait_s + request_s + parse_s
                        elif microbatch_generate_t > 0.0:
                            timing_out["t_total_file_s"] = microbatch_generate_t

                        prompt_result["timing"] = timing_out

                    prompt_result["candidate_order"] = int(row["candidate_order"])
                    prompt_result["microbatch_id"] = current_microbatch_id
                    prompt_results_by_doc.setdefault(int(row["original_index"]), []).append(prompt_result)

                infer_bar.update(len(batch_rows))
        finally:
            infer_bar.close()

        return prompt_results_by_doc, batch_generate_total_t

    def _aggregate_vllm_doc_timing(self, doc_plan: Dict[str, Any], prompt_results: List[Dict[str, Any]]) -> Dict[str, float]:
        if self.timings == "off":
            return {}

        timing_out: Dict[str, float] = {}
        batch_generate_by_id: Dict[int, float] = {}
        summed_keys = (
            "t_vllm_request_s",
            "t_vllm_ttft_s",
            "t_vllm_decode_s",
            "t_vllm_queue_s",
            "t_vllm_scheduler_s",
            "t_vllm_model_forward_s",
            "t_vllm_model_execute_s",
            "t_prompt_s",
            "t_parse_s",
            "t_prebatch_wait_s",
        )

        for prompt_result in prompt_results:
            timing = prompt_result.get("timing") or {}
            microbatch_id = prompt_result.get("microbatch_id")
            batch_generate = timing.get("t_batch_generate_s")
            if microbatch_id is not None and isinstance(batch_generate, (int, float)):
                batch_generate_by_id[int(microbatch_id)] = max(
                    batch_generate_by_id.get(int(microbatch_id), 0.0),
                    max(0.0, float(batch_generate)),
                )
            for key in summed_keys:
                value = timing.get(key)
                if isinstance(value, (int, float)):
                    timing_out[key] = timing_out.get(key, 0.0) + max(0.0, float(value))

        if batch_generate_by_id:
            timing_out["t_batch_generate_s"] = sum(batch_generate_by_id.values())

        if self.timings == "detailed":
            t_read0 = doc_plan.get("t_read0")
            t_read1 = doc_plan.get("t_read1")
            if t_read0 is not None and t_read1 is not None:
                timing_out["t_read_s"] = max(0.0, t_read1 - t_read0)

        read_s = float(timing_out.get("t_read_s", 0.0))
        prompt_s = float(timing_out.get("t_prompt_s", 0.0))
        prebatch_wait_s = float(timing_out.get("t_prebatch_wait_s", 0.0))
        request_s = float(timing_out.get("t_vllm_request_s", 0.0))
        parse_s = float(timing_out.get("t_parse_s", 0.0))
        total_file_s = read_s + prompt_s + prebatch_wait_s + request_s + parse_s
        if total_file_s > 0.0:
            timing_out["t_total_file_s"] = total_file_s
        elif "t_batch_generate_s" in timing_out:
            timing_out["t_total_file_s"] = float(timing_out["t_batch_generate_s"])

        return timing_out

    def _run_vllm_generate_plan(
        self,
        doc_rows: List[Dict[str, Any]],
        *,
        prep_desc: str,
        infer_desc: str,
        force_chunked: bool,
        merge_policy: Optional[str],
        prompt_suffix_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        do_sample_override: Optional[bool] = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], float]:
        prompt_rows, doc_plans = self._prepare_vllm_prompt_rows(
            doc_rows,
            prep_desc=prep_desc,
            force_chunked=force_chunked,
            merge_policy=merge_policy,
            prompt_suffix_override=prompt_suffix_override,
        )
        prompt_results_by_doc, batch_generate_total_t = self._run_vllm_prompt_batches(
            prompt_rows,
            infer_desc=infer_desc,
            temperature_override=temperature_override,
            do_sample_override=do_sample_override,
        )

        results: Dict[int, Dict[str, Any]] = {}
        for original_index, doc_plan in doc_plans.items():
            prompt_results = sorted(prompt_results_by_doc.get(original_index, []), key=lambda item: int(item.get("candidate_order", 0)))
            candidates: List[PatentMetadata] = []
            raw_outputs: List[Any] = []
            errors: List[Dict[str, Any]] = []
            for prompt_result in prompt_results:
                metadata = prompt_result.get("metadata")
                if isinstance(metadata, PatentMetadata):
                    candidates.append(metadata)
                    raw_outputs.append(prompt_result.get("raw_output"))
                else:
                    errors.append(prompt_result)

            timing = self._aggregate_vllm_doc_timing(doc_plan, prompt_results)
            if not candidates:
                if errors:
                    first_error = errors[0]
                    results[original_index] = {
                        **doc_plan,
                        "error": first_error.get("error", "exception: RuntimeError"),
                        "error_type": first_error.get("error_type", "RuntimeError"),
                        "error_detail": first_error.get("error_detail", "vLLM generation failed"),
                        "timing": timing,
                    }
                else:
                    results[original_index] = {
                        **doc_plan,
                        "error": "exception: RuntimeError",
                        "error_type": "RuntimeError",
                        "error_detail": "Missing vLLM outputs for document",
                        "timing": timing,
                    }
                continue

            if doc_plan["use_chunked"]:
                metadata = self._merge_metadata_candidates(candidates, policy=merge_policy)
                raw_output: Any = raw_outputs if self.save_raw_output else None
            else:
                metadata = candidates[0]
                raw_output = raw_outputs[0] if raw_outputs else None

            results[original_index] = {**doc_plan, "metadata": metadata, "raw_output": raw_output, "timing": timing}

        return results, batch_generate_total_t

    def _run_vllm_batched_baseline_strategy(self, doc_rows: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], float]:
        base_results, batch_generate_total_t = self._run_vllm_generate_plan(
            doc_rows,
            prep_desc="🧠 Préparation prompts vLLM",
            infer_desc="🚀 Inférence vLLM",
            force_chunked=False,
            merge_policy=None,
        )
        results: Dict[int, Dict[str, Any]] = {}
        for original_index, result in base_results.items():
            if "metadata" not in result:
                results[original_index] = result
                continue
            results[original_index] = {
                **result,
                "strategy_meta": {
                    "strategy_used": "baseline",
                    "was_rerun": False,
                    "pass_count": result["pass_count"],
                    "chunks_count": result["chunks_count"],
                    "header_first_used": False,
                    "merge_policy_used": result["merge_policy_used"],
                    "self_consistency_n_used": 1,
                    "vllm_doc_batch_size_used": self.doc_batch_size or len(doc_rows),
                    "timing": result.get("timing") or {},
                },
            }
        return results, batch_generate_total_t

    def _run_vllm_batched_chunked_strategy(self, doc_rows: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], float]:
        base_results, batch_generate_total_t = self._run_vllm_generate_plan(
            doc_rows,
            prep_desc="🧠 Préparation chunks vLLM",
            infer_desc="🚀 Inférence vLLM (chunked)",
            force_chunked=True,
            merge_policy=self.merge_policy,
        )
        results: Dict[int, Dict[str, Any]] = {}
        for original_index, result in base_results.items():
            if "metadata" not in result:
                results[original_index] = result
                continue
            results[original_index] = {
                **result,
                "strategy_meta": {
                    "strategy_used": "chunked",
                    "was_rerun": False,
                    "pass_count": result["pass_count"],
                    "chunks_count": result["chunks_count"],
                    "header_first_used": False,
                    "merge_policy_used": self.merge_policy,
                    "self_consistency_n_used": 1,
                    "vllm_doc_batch_size_used": self.doc_batch_size or len(doc_rows),
                    "timing": result.get("timing") or {},
                },
            }
        return results, batch_generate_total_t

    def _run_vllm_batched_header_first_strategy(self, doc_rows: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], float]:
        header_rows = []
        for doc_row in doc_rows:
            lines = str(doc_row["ocr_text"]).splitlines()
            header_text = "\n".join(lines[: self.header_lines]) if lines else str(doc_row["ocr_text"])
            header_rows.append({**doc_row, "ocr_text": header_text})

        header_results, batch_generate_total_t = self._run_vllm_generate_plan(
            header_rows,
            prep_desc="🧠 Préparation headers vLLM",
            infer_desc="🚀 Inférence vLLM (header_first/header)",
            force_chunked=False,
            merge_policy=None,
        )

        fallback_doc_rows: List[Dict[str, Any]] = []
        missing_by_doc: Dict[int, List[str]] = {}
        for doc_row in doc_rows:
            original_index = int(doc_row["original_index"])
            header_result = header_results.get(original_index, {})
            header_meta = header_result.get("metadata") if isinstance(header_result.get("metadata"), PatentMetadata) else PatentMetadata()
            missing = self._missing_critical_fields(header_meta)
            missing_by_doc[original_index] = missing
            if missing:
                fallback_doc_rows.append(doc_row)

        full_results: Dict[int, Dict[str, Any]] = {}
        if fallback_doc_rows:
            rerun_results, rerun_generate_t = self._run_vllm_generate_plan(
                fallback_doc_rows,
                prep_desc="🧠 Préparation fallback vLLM",
                infer_desc="🚀 Inférence vLLM (header_first/full)",
                force_chunked=False,
                merge_policy=None,
            )
            full_results = rerun_results
            batch_generate_total_t += rerun_generate_t

        results: Dict[int, Dict[str, Any]] = {}
        for doc_row in doc_rows:
            original_index = int(doc_row["original_index"])
            header_result = header_results.get(original_index, {})
            header_ok = isinstance(header_result.get("metadata"), PatentMetadata)
            header_meta = header_result.get("metadata") if header_ok else PatentMetadata()
            header_raw = header_result.get("raw_output")
            missing = missing_by_doc.get(original_index, [])
            fallback_to_full = bool(missing)

            if fallback_to_full:
                full_result = full_results.get(original_index, {})
                full_ok = isinstance(full_result.get("metadata"), PatentMetadata)
                candidates: List[PatentMetadata] = []
                if header_ok:
                    candidates.append(header_meta)
                if full_ok:
                    candidates.append(full_result["metadata"])
                if not candidates:
                    error_result = full_result or header_result or {
                        **doc_row,
                        "error": "exception: RuntimeError",
                        "error_type": "RuntimeError",
                        "error_detail": "header_first failed in both passes",
                    }
                    results[original_index] = error_result
                    continue
                merged = self._merge_metadata_candidates(candidates, policy=self.merge_policy) if len(candidates) > 1 else candidates[0]
                timing: Dict[str, float] = {}
                if isinstance(header_result.get("timing"), dict):
                    timing.update({f"header_{key}": value for key, value in header_result["timing"].items()})
                if isinstance(full_result.get("timing"), dict):
                    timing.update({f"pass1_{key}": value for key, value in full_result["timing"].items()})
                pass_count = 1 + int(full_result.get("pass_count", 1))
                chunks_count = 1 + int(full_result.get("chunks_count", 1))
                raw_output = {"header": header_raw, "full_text": full_result.get("raw_output")}
            else:
                if not header_ok:
                    results[original_index] = header_result or {
                        **doc_row,
                        "error": "exception: RuntimeError",
                        "error_type": "RuntimeError",
                        "error_detail": "header_first header pass failed",
                    }
                    continue
                merged = header_meta
                timing = {f"header_{key}": value for key, value in (header_result.get("timing") or {}).items()}
                pass_count = 1
                chunks_count = 1
                raw_output = {"header": header_raw}

            results[original_index] = {
                **doc_row,
                "metadata": merged,
                "raw_output": raw_output,
                "strategy_meta": {
                    "strategy_used": "header_first",
                    "was_rerun": fallback_to_full,
                    "pass_count": pass_count,
                    "chunks_count": chunks_count,
                    "header_first_used": True,
                    "fallback_to_full": fallback_to_full,
                    "missing_critical_fields": missing,
                    "merge_policy_used": self.merge_policy,
                    "self_consistency_n_used": 1,
                    "vllm_doc_batch_size_used": self.doc_batch_size or len(doc_rows),
                    "timing": timing,
                },
            }
        return results, batch_generate_total_t

    def _run_vllm_batched_two_pass_targeted_strategy(self, doc_rows: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], float]:
        pass1_results, batch_generate_total_t = self._run_vllm_generate_plan(
            doc_rows,
            prep_desc="🧠 Préparation pass1 vLLM",
            infer_desc="🚀 Inférence vLLM (two_pass/pass1)",
            force_chunked=False,
            merge_policy=None,
        )

        rerun_doc_rows: List[Dict[str, Any]] = []
        pass1_confidence: Dict[int, float] = {}
        pass1_subscores: Dict[int, Dict[str, float]] = {}
        rerun_by_doc: Dict[int, bool] = {}
        for doc_row in doc_rows:
            original_index = int(doc_row["original_index"])
            pass1_result = pass1_results.get(original_index, {})
            if isinstance(pass1_result.get("metadata"), PatentMetadata):
                conf, subscores = self.compute_confidence(pass1_result["metadata"], str(doc_row["ocr_text"]))
            else:
                conf, subscores = 0.0, {}
            pass1_confidence[original_index] = conf
            pass1_subscores[original_index] = subscores
            was_rerun = conf < self.targeted_rerun_threshold
            rerun_by_doc[original_index] = was_rerun
            if was_rerun:
                rerun_doc_rows.append(doc_row)

        correction_suffix = (
            "\n\nCorrection mode: Re-check missing/uncertain fields and date consistency against the text. "
            "Output only corrected JSON.\n"
        )

        pass2_results: Dict[int, Dict[str, Any]] = {}
        if rerun_doc_rows:
            rerun_results, rerun_generate_t = self._run_vllm_generate_plan(
                rerun_doc_rows,
                prep_desc="🧠 Préparation pass2 vLLM",
                infer_desc="🚀 Inférence vLLM (two_pass/pass2)",
                force_chunked=False,
                merge_policy=None,
                prompt_suffix_override=self.prompt_suffix + correction_suffix,
            )
            pass2_results = rerun_results
            batch_generate_total_t += rerun_generate_t

        results: Dict[int, Dict[str, Any]] = {}
        for doc_row in doc_rows:
            original_index = int(doc_row["original_index"])
            pass1_result = pass1_results.get(original_index, {})
            pass1_ok = isinstance(pass1_result.get("metadata"), PatentMetadata)
            was_rerun = rerun_by_doc.get(original_index, False)

            if was_rerun:
                pass2_result = pass2_results.get(original_index, {})
                pass2_ok = isinstance(pass2_result.get("metadata"), PatentMetadata)
                candidates: List[PatentMetadata] = []
                if pass1_ok:
                    candidates.append(pass1_result["metadata"])
                if pass2_ok:
                    candidates.append(pass2_result["metadata"])
                if not candidates:
                    error_result = pass2_result or pass1_result or {
                        **doc_row,
                        "error": "exception: RuntimeError",
                        "error_type": "RuntimeError",
                        "error_detail": "two_pass_targeted failed in both passes",
                    }
                    results[original_index] = error_result
                    continue
                merged = self._merge_metadata_candidates(candidates, policy=self.merge_policy) if len(candidates) > 1 else candidates[0]
                timing: Dict[str, float] = {}
                if isinstance(pass1_result.get("timing"), dict):
                    timing.update({f"pass1_{key}": value for key, value in pass1_result["timing"].items()})
                if isinstance(pass2_result.get("timing"), dict):
                    timing.update({f"pass2_{key}": value for key, value in pass2_result["timing"].items()})
                pass_count = int(pass1_result.get("pass_count", 1)) + int(pass2_result.get("pass_count", 1))
                chunks_count = int(pass1_result.get("chunks_count", 1)) + int(pass2_result.get("chunks_count", 1))
                raw_output = {"pass1": pass1_result.get("raw_output"), "pass2": pass2_result.get("raw_output")}
            else:
                if not pass1_ok:
                    results[original_index] = pass1_result or {
                        **doc_row,
                        "error": "exception: RuntimeError",
                        "error_type": "RuntimeError",
                        "error_detail": "two_pass_targeted pass1 failed",
                    }
                    continue
                merged = pass1_result["metadata"]
                timing = {f"pass1_{key}": value for key, value in (pass1_result.get("timing") or {}).items()}
                pass_count = int(pass1_result.get("pass_count", 1))
                chunks_count = int(pass1_result.get("chunks_count", 1))
                raw_output = {"pass1": pass1_result.get("raw_output")}

            results[original_index] = {
                **doc_row,
                "metadata": merged,
                "raw_output": raw_output,
                "strategy_meta": {
                    "strategy_used": "two_pass_targeted",
                    "was_rerun": was_rerun,
                    "targeted_rerun_threshold": self.targeted_rerun_threshold,
                    "pass1_confidence": pass1_confidence.get(original_index, 0.0),
                    "pass1_confidence_subscores": pass1_subscores.get(original_index, {}),
                    "pass_count": pass_count,
                    "chunks_count": chunks_count,
                    "header_first_used": False,
                    "merge_policy_used": self.merge_policy,
                    "self_consistency_n_used": 1,
                    "vllm_doc_batch_size_used": self.doc_batch_size or len(doc_rows),
                    "timing": timing,
                },
            }
        return results, batch_generate_total_t

    def _run_vllm_batched_self_consistency_strategy(self, doc_rows: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], float]:
        n = max(1, self.self_consistency_n)
        pass_results_list: List[Dict[int, Dict[str, Any]]] = []
        batch_generate_total_t = 0.0
        for pass_index in range(n):
            pass_results, pass_generate_t = self._run_vllm_generate_plan(
                doc_rows,
                prep_desc=f"🧠 Préparation SC {pass_index + 1}/{n} vLLM",
                infer_desc=f"🚀 Inférence vLLM (self_consistency {pass_index + 1}/{n})",
                force_chunked=False,
                merge_policy=None,
                temperature_override=self.self_consistency_temp,
                do_sample_override=(self.self_consistency_temp > 0.0 or self.do_sample),
            )
            pass_results_list.append(pass_results)
            batch_generate_total_t += pass_generate_t

        results: Dict[int, Dict[str, Any]] = {}
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
        for doc_row in doc_rows:
            original_index = int(doc_row["original_index"])
            candidates: List[PatentMetadata] = []
            all_timings: List[Dict[str, float]] = []
            all_raw: List[Any] = []
            pass_count = 0
            chunks_count = 0
            first_error: Optional[Dict[str, Any]] = None

            for pass_results in pass_results_list:
                pass_result = pass_results.get(original_index, {})
                if isinstance(pass_result.get("metadata"), PatentMetadata):
                    candidates.append(pass_result["metadata"])
                    all_raw.append(pass_result.get("raw_output"))
                elif first_error is None and pass_result:
                    first_error = pass_result
                timing = pass_result.get("timing")
                if isinstance(timing, dict):
                    all_timings.append(timing)
                pass_count += int(pass_result.get("pass_count", 1))
                chunks_count += int(pass_result.get("chunks_count", 1))

            if not candidates:
                results[original_index] = first_error or {
                    **doc_row,
                    "error": "exception: RuntimeError",
                    "error_type": "RuntimeError",
                    "error_detail": "self_consistency failed for all passes",
                }
                continue

            t_merge0 = time.perf_counter() if self.timings != "off" else None
            merged = self._merge_metadata_candidates(candidates, policy="vote_majority", fallback_policy="prefer_non_null")
            t_merge1 = time.perf_counter() if self.timings != "off" else None

            agg_timing: Dict[str, float] = {}
            for timing in all_timings:
                for key, value in timing.items():
                    if isinstance(value, (int, float)):
                        agg_timing[key] = agg_timing.get(key, 0.0) + float(value)
            if t_merge0 is not None and t_merge1 is not None:
                agg_timing["t_self_consistency_merge_s"] = max(0.0, t_merge1 - t_merge0)

            rows = [candidate.model_dump(mode="json") for candidate in candidates]
            variance_by_field = {field: round(self._field_variance(rows, field), 6) for field in fields}
            variance_mean = round(sum(variance_by_field.values()) / len(variance_by_field), 6)

            results[original_index] = {
                **doc_row,
                "metadata": merged,
                "raw_output": all_raw,
                "strategy_meta": {
                    "strategy_used": "self_consistency",
                    "was_rerun": n > 1,
                    "pass_count": pass_count,
                    "chunks_count": chunks_count,
                    "header_first_used": False,
                    "merge_policy_used": "vote_majority",
                    "self_consistency_n_used": n,
                    "self_consistency_temp_used": self.self_consistency_temp,
                    "self_consistency_variance": variance_mean,
                    "self_consistency_variance_by_field": variance_by_field,
                    "vllm_doc_batch_size_used": self.doc_batch_size or len(doc_rows),
                    "timing": agg_timing,
                },
            }
        return results, batch_generate_total_t

    def _run_vllm_batched_strategy(self, doc_rows: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], float]:
        if self.strategy == "baseline":
            return self._run_vllm_batched_baseline_strategy(doc_rows)
        if self.strategy == "chunked":
            return self._run_vllm_batched_chunked_strategy(doc_rows)
        if self.strategy == "header_first":
            return self._run_vllm_batched_header_first_strategy(doc_rows)
        if self.strategy == "two_pass_targeted":
            return self._run_vllm_batched_two_pass_targeted_strategy(doc_rows)
        if self.strategy == "self_consistency":
            return self._run_vllm_batched_self_consistency_strategy(doc_rows)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _batch_extract_vllm(self, txt_dir: Path, out_file: Path, limit: Optional[int] = None, raw_output_dir: Optional[Path] = None) -> int:
        txt_files = sorted(txt_dir.glob("*.txt"))
        total = len(txt_files)
        if limit is not None and limit < total:
            txt_files = txt_files[:limit]
            print(
                f"⚙️ Limitation vLLM à {limit} documents (sur {total} total)"
                f" | strategy={self.strategy}"
                f" | micro_batch={self.doc_batch_size or 'all'}"
            )
        else:
            print(
                f"⚙️ Traitement vLLM de {len(txt_files)} documents"
                f" | strategy={self.strategy}"
                f" | micro_batch={self.doc_batch_size or 'all'}"
                f" | sort_by_prompt_len={'yes' if self.sort_by_prompt_length else 'no'}"
            )

        out_file.parent.mkdir(parents=True, exist_ok=True)
        if raw_output_dir is not None:
            raw_output_dir.mkdir(parents=True, exist_ok=True)

        t_batch0 = time.perf_counter() if self.timings != "off" else None
        doc_rows, records_by_index = self._load_vllm_doc_rows(txt_files)
        doc_rows_by_index = {int(doc_row["original_index"]): doc_row for doc_row in doc_rows}
        strategy_results, batch_generate_total_t = self._run_vllm_batched_strategy(doc_rows)

        with out_file.open("w", encoding="utf-8") as f_out:
            count = 0
            for index in range(len(txt_files)):
                record = records_by_index.get(index)
                if record is None:
                    result = strategy_results.get(index)
                    doc_row = doc_rows_by_index.get(index)
                    if result is None or doc_row is None:
                        raise RuntimeError(f"Missing vLLM record for document index {index}")
                    if "metadata" in result:
                        strategy_meta = dict(result.get("strategy_meta") or {})
                        confidence, subscores = self.compute_confidence(result["metadata"], str(doc_row["ocr_text"]))
                        strategy_meta["confidence_score"] = confidence
                        strategy_meta["confidence_subscores"] = subscores
                        record = self._build_success_record(
                            doc_row,
                            metadata=result["metadata"],
                            strategy_meta=strategy_meta,
                            raw_output=result.get("raw_output"),
                            raw_output_dir=raw_output_dir,
                        )
                    else:
                        record = self._make_exception_record(
                            doc_row["txt_path"],
                            t_file0=doc_row.get("t_file0"),
                            error_type=result.get("error_type", "RuntimeError"),
                            error_detail=result.get("error_detail", "vLLM generation failed"),
                            timing=result.get("timing"),
                        )
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        if self.timings != "off" and t_batch0 is not None:
            total_elapsed = max(0.0, time.perf_counter() - t_batch0)
            print(f"⏱️  vLLM batch elapsed: {total_elapsed:.3f}s (generate={batch_generate_total_t:.3f}s)")
        print(f"✅ Extraction vLLM complète → {count} documents traités")
        print(f"📊 Résultats: {out_file}")
        return count


class PatentExtractionRunner:
    """Public runner façade for the standalone vLLM-only package."""

    def __init__(self, profile: ProfileConfig):
        self.profile = profile
        self._prompt_text: Optional[str] = None
        self._extractor: Optional[PatentExtractor] = None

    @classmethod
    def from_profile(
        cls,
        *,
        name: str = DEFAULT_PROFILE_NAME,
        profile_path: Optional[str | Path] = None,
        prompt_path: Optional[str | Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PatentExtractionRunner":
        profile = load_profile(
            name=name,
            profile_path=profile_path,
            prompt_path=prompt_path,
            overrides=overrides,
        )
        return cls(profile=profile)

    @property
    def prompt_text(self) -> Optional[str]:
        if self._prompt_text is None:
            self._prompt_text = self.profile.read_prompt_text()
        return self._prompt_text

    def build_extractor(self) -> PatentExtractor:
        if self._extractor is None:
            extraction = self.profile.extraction
            self._extractor = PatentExtractor(
                model_name=extraction.model_name,
                torch_dtype=extraction.torch_dtype,
                quantization=extraction.vllm.quantization,
                prompt_id=extraction.prompt_id,
                prompt_template=self.prompt_text,
                guardrail_profile=extraction.guardrail_profile,
                max_ocr_chars=extraction.max_ocr_chars,
                max_new_tokens=extraction.max_new_tokens,
                temperature=extraction.temperature,
                do_sample=extraction.do_sample,
                extraction_mode=extraction.strategy.extraction_mode,
                chunk_size_chars=extraction.strategy.chunk_size_chars,
                chunk_overlap_chars=extraction.strategy.chunk_overlap_chars,
                extraction_passes=extraction.strategy.extraction_passes,
                strategy=extraction.strategy.name,
                header_lines=extraction.strategy.header_lines,
                targeted_rerun_threshold=extraction.strategy.targeted_rerun_threshold,
                self_consistency_n=extraction.strategy.self_consistency_n,
                self_consistency_temp=extraction.strategy.self_consistency_temp,
                merge_policy=extraction.strategy.merge_policy,
                save_strategy_meta=extraction.save_strategy_meta,
                save_raw_output=extraction.save_raw_output,
                timings=extraction.timings,
                enable_prefix_caching=extraction.vllm.enable_prefix_caching,
                tensor_parallel_size=extraction.vllm.tensor_parallel_size,
                gpu_memory_utilization=extraction.vllm.gpu_memory_utilization,
                max_model_len=extraction.vllm.max_model_len,
                swap_space=extraction.vllm.swap_space,
                enforce_eager=extraction.vllm.enforce_eager,
                doc_batch_size=extraction.vllm.doc_batch_size,
                sort_by_prompt_length=extraction.vllm.sort_by_prompt_length,
                tokenizer_mode=extraction.vllm.tokenizer_mode,
            )
        return self._extractor

    def extract(self, ocr_text: str, *, debug: bool = False):
        return self.build_extractor().extract(ocr_text, debug=debug)

    def extract_file(self, txt_path: str | Path, *, raw_output_dir: Optional[str | Path] = None):
        raw_dir = Path(raw_output_dir) if raw_output_dir is not None else None
        return self.build_extractor().extract_from_file(Path(txt_path), raw_output_dir=raw_dir)

    def batch_extract(
        self,
        *,
        texts_dir: str | Path,
        out_root: str | Path,
        run_name: str,
        limit: Optional[int] = None,
        force: bool = False,
    ) -> RunArtifacts:
        texts_path = Path(texts_dir)
        if not texts_path.exists():
            raise FileNotFoundError(f"Missing texts dir: {texts_path}")
        if not texts_path.is_dir():
            raise NotADirectoryError(f"texts_dir must be a directory: {texts_path}")

        out_root_path = Path(out_root)
        run_dir = out_root_path / run_name
        preds_path = run_dir / "preds.jsonl"
        run_meta_path = run_dir / "run.json"
        raw_outputs_dir = run_dir / "raw_outputs" if self.profile.extraction.save_raw_output else None

        if run_dir.exists() and not force:
            raise FileExistsError(f"Run dir already exists: {run_dir}. Pass force=True to overwrite outputs.")
        run_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(texts_path.glob("*.txt"))
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        docs_written = self.build_extractor().batch_extract(
            txt_dir=texts_path,
            out_file=preds_path,
            limit=limit,
            raw_output_dir=raw_outputs_dir,
        )
        wall_s = max(0.0, time.perf_counter() - t0)
        finished_at = datetime.now(timezone.utc)

        prompt_text = self.prompt_text or ""
        run_meta = {
            "runner": "patent_extraction",
            "profile_name": self.profile.name,
            "profile_path": str(self.profile.definition_path) if self.profile.definition_path is not None else None,
            "description": self.profile.description,
            "texts_dir": str(texts_path),
            "run_name": run_name,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "wall_s": round(wall_s, 6),
            "docs_total": len(txt_files),
            "docs_limit": limit,
            "docs_written": docs_written,
            "preds_path": str(preds_path),
            "raw_outputs_dir": str(raw_outputs_dir) if raw_outputs_dir is not None else None,
            "prompt_path": str(self.profile.extraction.prompt_path) if self.profile.extraction.prompt_path else None,
            "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest() if prompt_text else None,
            "config": _jsonable(self.profile.extraction.as_dict()),
        }
        run_meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        return RunArtifacts(
            run_dir=run_dir,
            preds_path=preds_path,
            run_meta_path=run_meta_path,
            raw_outputs_dir=raw_outputs_dir,
            docs_written=docs_written,
            wall_s=wall_s,
        )


__all__ = ["PatentExtractionRunner", "PatentExtractor", "RunArtifacts", "_PROMPT_BY_ID"]
