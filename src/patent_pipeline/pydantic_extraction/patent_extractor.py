# 📄 src/patent_pipeline/pydantic_extraction/patent_extractor.py
"""
PatentExtractor

Façade publique pour l'extraction SLM.

Les responsabilités internes sont maintenant réparties par domaine:
- runtime.py      : chargement modèle + génération
- postprocess.py  : extraction JSON, normalisation, merge, confidence
- strategies.py   : orchestration single/chunked/header-first/etc.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from tqdm import tqdm

from .models import PatentExtraction, PatentMetadata
from .prompt_templates import PROMPT_EXTRACTION_V1, PROMPT_EXTRACTION_V2, PROMPT_EXTRACTION_V3
from . import postprocess as postprocess_mod
from . import runtime as runtime_mod
from . import strategies as strategy_mod
from ..utils.device_utils import get_device

# Compatibility re-exports used by tests and monkeypatching.
AutoModelForCausalLM = runtime_mod.AutoModelForCausalLM
BitsAndBytesConfig = runtime_mod.BitsAndBytesConfig
Mistral3ForCausalLM = runtime_mod.Mistral3ForCausalLM
Mistral3ForConditionalGeneration = runtime_mod.Mistral3ForConditionalGeneration
_HAS_MLX = runtime_mod.HAS_MLX


_PROMPT_BY_ID: Dict[str, str] = {
    "v1": PROMPT_EXTRACTION_V1,
    "v2": PROMPT_EXTRACTION_V2,
    "v3": PROMPT_EXTRACTION_V3,
}

_JSON_ONLY_SUFFIX = "\n\nNow output ONLY the JSON object, without any extra text.\n"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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


class PatentExtractor:
    """
    Extracteur de métadonnées de brevets à partir de textes OCR.

    Supporte:
    - MLX (Apple Silicon)
    - PyTorch (CPU/CUDA)
    """

    _config_name = staticmethod(runtime_mod.config_name)
    _config_model_type = staticmethod(runtime_mod.config_model_type)
    _resolve_torch_dtype = runtime_mod.resolve_torch_dtype
    _resolve_attn_implementation = staticmethod(runtime_mod.resolve_attn_implementation)
    _build_quantization_config = staticmethod(runtime_mod.build_quantization_config)
    _resolve_cache_implementation = staticmethod(runtime_mod.resolve_cache_implementation)
    _build_generate_kwargs = staticmethod(runtime_mod.build_generate_kwargs)
    _resolve_vllm_dtype = staticmethod(runtime_mod.resolve_vllm_dtype)
    _build_vllm_sampling_params = staticmethod(runtime_mod.build_vllm_sampling_params)
    _is_mistral31_model = runtime_mod.is_mistral31_model
    _load_tokenizer = runtime_mod.load_tokenizer
    _generate = runtime_mod.generate_text

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
    _is_company_name = staticmethod(postprocess_mod.is_company_name)
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

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: Literal["auto", "mlx", "pytorch", "vllm"] = "auto",
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        torch_dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto",
        quantization: Literal["none", "bnb_8bit", "bnb_4bit"] = "none",
        attn_implementation: Literal["auto", "sdpa", "flash_attention_2"] = "auto",
        cache_implementation: Literal["auto", "dynamic", "static", "offloaded", "offloaded_static"] = "auto",
        vllm_enable_prefix_caching: bool = False,
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_max_model_len: Optional[int] = None,
        vllm_swap_space: float = 4.0,
        vllm_enforce_eager: bool = False,
        vllm_doc_batch_size: Optional[int] = 32,
        vllm_sort_by_prompt_length: bool = True,
        vllm_tokenizer_mode: Literal["auto", "mistral"] = "auto",
        prompt_id: Optional[str] = None,
        prompt_template: Optional[str] = None,
        max_ocr_chars: int = 10000,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        extraction_mode: Literal["single", "chunked", "auto"] = "auto",
        chunk_size_chars: int = 7000,
        chunk_overlap_chars: int = 800,
        extraction_passes: int = 2,
        strategy: Literal[
            "baseline",
            "chunked",
            "header_first",
            "two_pass_targeted",
            "self_consistency",
        ] = "baseline",
        header_lines: int = 30,
        targeted_rerun_threshold: float = 0.6,
        self_consistency_n: int = 2,
        self_consistency_temp: float = 0.2,
        merge_policy: Literal["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"] = "prefer_non_null",
        save_strategy_meta: bool = False,
        save_raw_output: bool = False,
        timings: Literal["off", "basic", "detailed"] = "off",
    ):
        self.model_name = model_name or os.getenv("HF_MODEL", "mlx-community/Mistral-7B-Instruct-v0.3")
        self.max_ocr_chars = max_ocr_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.extraction_mode = extraction_mode
        self.chunk_size_chars = int(chunk_size_chars)
        self.chunk_overlap_chars = int(chunk_overlap_chars)
        self.extraction_passes = int(extraction_passes)
        self.strategy = strategy
        self.header_lines = int(header_lines)
        self.targeted_rerun_threshold = float(targeted_rerun_threshold)
        self.self_consistency_n = int(self_consistency_n)
        self.self_consistency_temp = float(self_consistency_temp)
        self.merge_policy = merge_policy
        self.save_strategy_meta = bool(save_strategy_meta)
        self.save_raw_output = bool(save_raw_output)
        self.timings = timings
        self.torch_dtype = str(torch_dtype).strip().lower()
        self.quantization = str(quantization).strip().lower()
        self.attn_implementation = str(attn_implementation).strip().lower()
        self.cache_implementation = str(cache_implementation).strip().lower()
        self.vllm_enable_prefix_caching = bool(vllm_enable_prefix_caching)
        self.vllm_tensor_parallel_size = int(vllm_tensor_parallel_size)
        self.vllm_gpu_memory_utilization = float(vllm_gpu_memory_utilization)
        self.vllm_max_model_len = None if vllm_max_model_len is None else int(vllm_max_model_len)
        self.vllm_swap_space = float(vllm_swap_space)
        self.vllm_enforce_eager = bool(vllm_enforce_eager)
        self.vllm_doc_batch_size = None if vllm_doc_batch_size is None else int(vllm_doc_batch_size)
        self.vllm_sort_by_prompt_length = bool(vllm_sort_by_prompt_length)
        self.vllm_tokenizer_mode = str(vllm_tokenizer_mode).strip().lower()

        if self.torch_dtype not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("torch_dtype must be one of: auto|bf16|fp16|fp32")
        if self.quantization not in {"none", "bnb_8bit", "bnb_4bit"}:
            raise ValueError("quantization must be one of: none|bnb_8bit|bnb_4bit")
        if self.attn_implementation not in {"auto", "sdpa", "flash_attention_2"}:
            raise ValueError("attn_implementation must be one of: auto|sdpa|flash_attention_2")
        if self.cache_implementation not in {"auto", "dynamic", "static", "offloaded", "offloaded_static"}:
            raise ValueError(
                "cache_implementation must be one of: auto|dynamic|static|offloaded|offloaded_static"
            )
        if self.vllm_tensor_parallel_size < 1:
            raise ValueError("vllm_tensor_parallel_size must be >= 1")
        if not (0.0 < self.vllm_gpu_memory_utilization <= 1.0):
            raise ValueError("vllm_gpu_memory_utilization must be in (0, 1]")
        if self.vllm_max_model_len is not None and self.vllm_max_model_len < 1:
            raise ValueError("vllm_max_model_len must be >= 1")
        if self.vllm_swap_space < 0.0:
            raise ValueError("vllm_swap_space must be >= 0")
        if self.vllm_doc_batch_size is not None and self.vllm_doc_batch_size < 1:
            raise ValueError("vllm_doc_batch_size must be >= 1")
        if self.vllm_tokenizer_mode not in {"auto", "mistral"}:
            raise ValueError("vllm_tokenizer_mode must be one of: auto|mistral")
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

        self.prompt_suffix = _JSON_ONLY_SUFFIX
        self.prompt_id = prompt_id
        if self.prompt_id is not None:
            if self.prompt_id not in _PROMPT_BY_ID:
                raise ValueError(f"Unknown prompt_id={self.prompt_id!r}. Allowed: {sorted(_PROMPT_BY_ID.keys())}")
            self.prompt_template = _PROMPT_BY_ID[self.prompt_id]
            self.prompt_template_source = f"prompt_id:{self.prompt_id}"
        else:
            self.prompt_template = prompt_template or PROMPT_EXTRACTION_V2
            self.prompt_template_source = "inline_template" if prompt_template else "default:v2"

        if "{text}" not in self.prompt_template:
            raise ValueError("Le template doit contenir le placeholder {text}")
        self.prompt_hash = _sha256(self.prompt_template + self.prompt_suffix)

        if backend == "auto":
            self.backend = "mlx" if _HAS_MLX else "pytorch"
        else:
            self.backend = backend
            if self.backend == "mlx" and not _HAS_MLX:
                raise ImportError("MLX n'est pas installé. Installe avec: pip install mlx-lm")
        if self.backend == "vllm":
            if self.quantization != "none":
                raise ValueError("quantization is not yet supported with backend='vllm' in this extractor")
            if self.attn_implementation != "auto":
                raise ValueError("attn_implementation is not supported with backend='vllm'")
            if self.cache_implementation != "auto":
                raise ValueError("cache_implementation is not supported with backend='vllm'")
        elif self.backend != "pytorch":
            if self.quantization != "none":
                raise ValueError("quantization is only supported with backend='pytorch'")
            if self.attn_implementation != "auto":
                raise ValueError("attn_implementation is only supported with backend='pytorch'")
            if self.cache_implementation != "auto":
                raise ValueError("cache_implementation is only supported with backend='pytorch'")

        self.device = device or get_device()
        self.device_map = device_map or ("cuda" if self.device == "cuda" else "cpu")
        if self.backend == "pytorch" and self.device == "mps":
            print("⚠️  MPS backend instable → fallback CPU")
            self.device = "cpu"
            self.device_map = "cpu"

        self._last_timing: Optional[Dict[str, float]] = None
        self._last_raw_output: Optional[Any] = None
        self._last_strategy_meta: Optional[Dict[str, Any]] = None

        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.pipeline_task: Optional[str] = None
        self._debug_gen_logged = False

        self._load_model()

    @staticmethod
    def _resolve_model_route(config: Optional[Any]) -> Dict[str, Any]:
        return runtime_mod.resolve_model_route(
            config,
            auto_model_cls=AutoModelForCausalLM,
            mistral3_causal_cls=Mistral3ForCausalLM,
            mistral3_conditional_cls=Mistral3ForConditionalGeneration,
        )

    def _load_model(self) -> None:
        runtime_mod.load_model_into(
            self,
            resolve_model_route_fn=self._resolve_model_route,
            resolve_torch_dtype_fn=self._resolve_torch_dtype,
            load_tokenizer_fn=self._load_tokenizer,
        )

    def set_prompt_template(self, template: str) -> None:
        if "{text}" not in template:
            raise ValueError("Le template doit contenir le placeholder {text}")
        self.prompt_id = None
        self.prompt_template = template
        self.prompt_template_source = "inline_template"
        self.prompt_hash = _sha256(self.prompt_template + self.prompt_suffix)

    def extract(self, ocr_text: str, debug: bool = False) -> PatentExtraction:
        try:
            metadata, strategy_meta, raw_output = self._run_strategy(ocr_text, debug=debug)
        except Exception as e:
            print(f"⚠️ Strategy '{self.strategy}' failed ({e.__class__.__name__}). Fallback to baseline.")
            metadata, timing, raw_output, base_meta = self._run_baseline_strategy(ocr_text, debug=debug)
            strategy_meta = {
                **base_meta,
                "strategy_used": "baseline",
                "strategy_fallback_from": self.strategy,
                "timing": timing or {},
            }

        confidence, subscores = self.compute_confidence(metadata, ocr_text)
        strategy_meta["confidence_score"] = confidence
        strategy_meta["confidence_subscores"] = subscores

        self._last_timing = strategy_meta.get("timing") or None
        self._last_raw_output = raw_output
        self._last_strategy_meta = strategy_meta
        return PatentExtraction(
            ocr_text=ocr_text,
            model=self.model_name,
            prediction=metadata,
        )

    def extract_from_file(self, txt_path: Path, raw_output_dir: Optional[Path] = None) -> dict:
        t_file0 = time.perf_counter() if self.timings != "off" else None

        try:
            t_read0 = time.perf_counter() if self.timings == "detailed" else None
            ocr_text = txt_path.read_text(encoding="utf-8", errors="ignore")
            t_read1 = time.perf_counter() if self.timings == "detailed" else None

            if not ocr_text.strip():
                rec = {
                    "file_name": txt_path.name,
                    "ocr_path": str(txt_path),
                    "error": "empty_ocr",
                    "strategy_used": self.strategy,
                    "confidence_score": 0.0,
                }
                if self.prompt_id is not None:
                    rec["prompt_id"] = self.prompt_id
                rec["prompt_hash"] = self.prompt_hash

                if self.timings != "off" and t_file0 is not None:
                    timing = {"t_total_file_s": time.perf_counter() - t_file0}
                    if self.timings == "detailed" and t_read0 is not None and t_read1 is not None:
                        timing["t_read_s"] = max(0.0, t_read1 - t_read0)
                    rec["timing"] = timing

                return rec

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

        except Exception as e:
            print(f"⚠️ Erreur sur {txt_path.name}: {e}")
            traceback.print_exc()

            rec = {
                "file_name": txt_path.name,
                "ocr_path": str(txt_path),
                "error": f"exception: {e.__class__.__name__}",
                "error_type": e.__class__.__name__,
                "error_detail": str(e),
                "strategy_used": self.strategy,
                "confidence_score": 0.0,
            }
            if self.prompt_id is not None:
                rec["prompt_id"] = self.prompt_id
            rec["prompt_hash"] = self.prompt_hash

            if self.timings != "off" and t_file0 is not None:
                rec["timing"] = {"t_total_file_s": max(0.0, time.perf_counter() - t_file0)}

            return rec

    def batch_extract(
        self,
        txt_dir: Path,
        out_file: Path,
        limit: Optional[int] = None,
        raw_output_dir: Optional[Path] = None,
    ) -> int:
        if self.backend == "vllm" and self.strategy == "baseline":
            return self._batch_extract_vllm_baseline(
                txt_dir=txt_dir,
                out_file=out_file,
                limit=limit,
                raw_output_dir=raw_output_dir,
            )

        txt_files = sorted(txt_dir.glob("*.txt"))
        total = len(txt_files)

        if limit is not None and limit < total:
            txt_files = txt_files[:limit]
            print(f"⚙️ Limitation à {limit} documents (sur {total} total)")
        else:
            print(f"⚙️ Traitement de {total} documents")

        out_file.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(out_file, "w", encoding="utf-8") as f_out:
            for txt_path in tqdm(txt_files, desc="🧠 Batch extraction", unit="doc"):
                record = self.extract_from_file(txt_path, raw_output_dir=raw_output_dir)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"✅ Extraction complète → {count} documents traités")
        print(f"📊 Résultats: {out_file}")

        return count

    def _batch_extract_vllm_baseline(
        self,
        txt_dir: Path,
        out_file: Path,
        limit: Optional[int] = None,
        raw_output_dir: Optional[Path] = None,
    ) -> int:
        txt_files = sorted(txt_dir.glob("*.txt"))
        total = len(txt_files)
        doc_batch_size = self.vllm_doc_batch_size

        if limit is not None and limit < total:
            txt_files = txt_files[:limit]
            print(
                f"⚙️ Limitation vLLM à {limit} documents (sur {total} total)"
                f" | micro_batch={doc_batch_size or 'all'}"
            )
        else:
            print(
                f"⚙️ Traitement vLLM de {total} documents"
                f" | micro_batch={doc_batch_size or 'all'}"
                f" | sort_by_prompt_len={'yes' if self.vllm_sort_by_prompt_length else 'no'}"
            )

        out_file.parent.mkdir(parents=True, exist_ok=True)
        if raw_output_dir is not None:
            raw_output_dir.mkdir(parents=True, exist_ok=True)

        records_by_index: Dict[int, Dict[str, Any]] = {}
        prompt_rows: List[Dict[str, Any]] = []
        t_batch0 = time.perf_counter() if self.timings != "off" else None
        batch_generate_total_t = 0.0

        for original_index, txt_path in enumerate(tqdm(txt_files, desc="🧠 Préparation prompts vLLM", unit="doc")):
            t_file0 = time.perf_counter() if self.timings != "off" else None
            t_read0 = time.perf_counter() if self.timings == "detailed" else None
            ocr_text = txt_path.read_text(encoding="utf-8", errors="ignore")
            t_read1 = time.perf_counter() if self.timings == "detailed" else None

            if not ocr_text.strip():
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
                    timing = {"t_total_file_s": time.perf_counter() - t_file0}
                    if self.timings == "detailed" and t_read0 is not None and t_read1 is not None:
                        timing["t_read_s"] = max(0.0, t_read1 - t_read0)
                    rec["timing"] = timing
                records_by_index[original_index] = rec
                continue

            t_prompt0 = time.perf_counter() if self.timings == "detailed" else None
            text_for_prompt = self._truncate_ocr(ocr_text)
            prompt = self.prompt_template.format(text=text_for_prompt) + self.prompt_suffix
            t_prompt1 = time.perf_counter() if self.timings == "detailed" else None
            prompt_rows.append(
                {
                    "txt_path": txt_path,
                    "ocr_text": ocr_text,
                    "prompt": prompt,
                    "t_file0": t_file0,
                    "t_read0": t_read0,
                    "t_read1": t_read1,
                    "t_prompt0": t_prompt0,
                    "t_prompt1": t_prompt1,
                    "original_index": original_index,
                }
            )

        work_rows = list(prompt_rows)
        if self.vllm_sort_by_prompt_length:
            work_rows.sort(key=lambda row: len(row["prompt"]), reverse=True)
        prompt_batches = _iter_batches(work_rows, doc_batch_size)

        infer_bar = tqdm(total=len(work_rows), desc="🚀 Inférence vLLM", unit="doc")
        try:
            for batch_rows in prompt_batches:
                prompts = [row["prompt"] for row in batch_rows]
                outputs: List[Any] = []
                microbatch_generate_t = 0.0
                t_gen_started_at: Optional[float] = None
                if prompts:
                    t_gen0 = time.perf_counter() if self.timings != "off" else None
                    t_gen_started_at = t_gen0
                    sampling_params = self._build_vllm_sampling_params(self)
                    outputs = list(self.model.generate(prompts, sampling_params=sampling_params, use_tqdm=False))
                    if t_gen0 is not None:
                        microbatch_generate_t = max(0.0, time.perf_counter() - t_gen0)
                        batch_generate_total_t += microbatch_generate_t

                for index, row in enumerate(batch_rows):
                    output = outputs[index] if index < len(outputs) else None
                    original_index = int(row["original_index"])
                    txt_path = row["txt_path"]
                    ocr_text = row["ocr_text"]
                    t_parse0 = time.perf_counter() if self.timings == "detailed" else None
                    t_parse1: Optional[float] = None
                    try:
                        if output is None:
                            raise RuntimeError("vLLM returned fewer outputs than prompts")
                        candidates = getattr(output, "outputs", None) or []
                        raw_output = (getattr(candidates[0], "text", None) or "") if candidates else ""
                        json_str = self._extract_json(raw_output)
                        metadata = self._parse_and_validate(json_str)
                        confidence, subscores = self.compute_confidence(metadata, ocr_text)

                        record = {
                            "ocr_text": ocr_text,
                            "model": self.model_name,
                            "prediction": metadata.model_dump(mode="json"),
                            "file_name": txt_path.name,
                            "ocr_path": str(txt_path),
                            "prompt_hash": self.prompt_hash,
                            "strategy_used": "baseline",
                            "confidence_score": confidence,
                        }
                        record["prediction"]["identifier"] = txt_path.stem.split("_")[0]
                        if self.prompt_id is not None:
                            record["prompt_id"] = self.prompt_id
                        if self.save_strategy_meta:
                            record["confidence_subscores"] = subscores
                            record["pass_count"] = 1
                            record["was_rerun"] = False
                            record["chunks_count"] = 1
                            record["header_first_used"] = False
                            record["merge_policy_used"] = self.merge_policy
                            record["self_consistency_n_used"] = 1
                            record["vllm_doc_batch_size_used"] = len(batch_rows)

                        if self.save_raw_output:
                            record["raw_output"] = raw_output
                            if raw_output_dir is not None:
                                raw_path = raw_output_dir / f"{txt_path.name}.raw.txt"
                                raw_path.write_text(raw_output, encoding="utf-8")
                                record["raw_output_path"] = str(raw_path)
                                record.pop("raw_output", None)

                    except Exception as e:
                        print(f"⚠️ Erreur vLLM sur {txt_path.name}: {e}")
                        traceback.print_exc()
                        record = {
                            "file_name": txt_path.name,
                            "ocr_path": str(txt_path),
                            "error": f"exception: {e.__class__.__name__}",
                            "error_type": e.__class__.__name__,
                            "error_detail": str(e),
                            "strategy_used": self.strategy,
                            "confidence_score": 0.0,
                            "prompt_hash": self.prompt_hash,
                        }
                        if self.prompt_id is not None:
                            record["prompt_id"] = self.prompt_id

                    if self.timings == "detailed":
                        t_parse1 = time.perf_counter()

                    if self.timings != "off":
                        timing_out: Dict[str, float] = {"t_batch_generate_s": microbatch_generate_t}
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
                        elif row.get("t_file0") is not None:
                            timing_out["t_total_file_s"] = max(0.0, time.perf_counter() - row["t_file0"])
                        record["timing"] = timing_out

                    records_by_index[original_index] = record
                infer_bar.update(len(batch_rows))
        finally:
            infer_bar.close()

        with open(out_file, "w", encoding="utf-8") as f_out:
            count = 0
            for index in range(len(txt_files)):
                record = records_by_index.get(index)
                if record is None:
                    raise RuntimeError(f"Missing vLLM record for document index {index}")
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        if self.timings != "off" and t_batch0 is not None:
            total_elapsed = max(0.0, time.perf_counter() - t_batch0)
            print(f"⏱️  vLLM batch elapsed: {total_elapsed:.3f}s (generate={batch_generate_total_t:.3f}s)")
        print(f"✅ Extraction vLLM complète → {count} documents traités")
        print(f"📊 Résultats: {out_file}")
        return count
