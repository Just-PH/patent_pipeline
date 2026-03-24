# 📄 src/patent_pipeline/pydantic_extraction/patent_extractor.py
"""
PatentExtractor

Rôle:
- Charger un modèle (MLX ou PyTorch/Transformers)
- Construire un prompt (via prompt_id v1/v2/v3 OU un template fourni)
- Générer une sortie
- Extraire un JSON du texte généré
- Normaliser + valider via Pydantic (PatentMetadata)
- (Option) mesurer des timings par document (off/basic/detailed)
- Écrire des records JSONL consommables par la suite (scoring)

Note importante (benchmark):
- prompt_hash doit être STABLE par run → on hash le TEMPLATE + suffix fixe,
  PAS le prompt final (qui inclut l'OCR et changerait par document).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import time
import traceback
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import regex as re
import torch
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import Mistral3ForCausalLM  # type: ignore
except Exception:
    Mistral3ForCausalLM = None  # type: ignore

try:
    from transformers import Mistral3ForConditionalGeneration  # type: ignore
except Exception:
    Mistral3ForConditionalGeneration = None  # type: ignore

from .models import PatentExtraction, PatentMetadata
from .prompt_templates import PROMPT_EXTRACTION_V1, PROMPT_EXTRACTION_V2, PROMPT_EXTRACTION_V3
from ..utils.device_utils import get_device

# Optional deps (MLX on Apple Silicon)
# Do not import mlx_lm at module import time: native init can crash on non-MLX hosts.
_HAS_MLX = importlib.util.find_spec("mlx_lm") is not None
mlx_lm = None


# ---------------------------------------------------------------------------
# Prompt registry (matrice 3D : prompts = [v1, v2, v3])
# ---------------------------------------------------------------------------
_PROMPT_BY_ID: Dict[str, str] = {
    "v1": PROMPT_EXTRACTION_V1,
    "v2": PROMPT_EXTRACTION_V2,
    "v3": PROMPT_EXTRACTION_V3,
}

_JSON_ONLY_SUFFIX = "\n\nNow output ONLY the JSON object, without any extra text.\n"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class PatentExtractor:
    """
    Extracteur de métadonnées de brevets à partir de textes OCR.

    Supporte:
    - MLX (Apple Silicon) : rapide en local dev sur Mac
    - PyTorch (CPU/CUDA)  : pour VM/H100

    La dimension "prompt" devient une vraie dimension de benchmark via:
    - prompt_id: "v1"/"v2"/"v3"
      OU
    - prompt_template: template string contenant "{text}"
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: Literal["auto", "mlx", "pytorch"] = "auto",
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        torch_dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto",
        # Prompt selection
        prompt_id: Optional[str] = None,
        prompt_template: Optional[str] = None,
        # Generation params
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
        # Timings
        timings: Literal["off", "basic", "detailed"] = "off",
    ):
        # ----------------------------
        # Basic config
        # ----------------------------
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

        if self.torch_dtype not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("torch_dtype must be one of: auto|bf16|fp16|fp32")

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

        # Constant suffix (part of prompt hash)
        self.prompt_suffix = _JSON_ONLY_SUFFIX

        # ----------------------------
        # Prompt config (matrice 3D)
        # ----------------------------
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

        # ✅ STABLE par run: template + suffix (pas le prompt rendu avec OCR)
        self.prompt_hash = _sha256(self.prompt_template + self.prompt_suffix)

        # ----------------------------
        # Backend selection
        # ----------------------------
        if backend == "auto":
            self.backend = "mlx" if _HAS_MLX else "pytorch"
        else:
            self.backend = backend
            if self.backend == "mlx" and not _HAS_MLX:
                raise ImportError("MLX n'est pas installé. Installe avec: pip install mlx-lm")

        # ----------------------------
        # Device selection (PyTorch)
        # ----------------------------
        self.device = device or get_device()
        self.device_map = device_map or ("cuda" if self.device == "cuda" else "cpu")

        # NOTE: MPS est souvent instable pour Transformers “text-generation”.
        if self.backend == "pytorch" and self.device == "mps":
            print("⚠️  MPS backend instable → fallback CPU")
            self.device = "cpu"
            self.device_map = "cpu"

        # last per-document timings set by extract()
        self._last_timing: Optional[Dict[str, float]] = None
        # last raw output(s): str/list/dict depending on strategy
        self._last_raw_output: Optional[Any] = None
        # strategy-level debug/meta for the last extract()
        self._last_strategy_meta: Optional[Dict[str, Any]] = None

        # Model objects
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.pipeline_task: Optional[str] = None
        self._debug_gen_logged = False

        # ----------------------------
        # Load model
        # ----------------------------
        self._load_model()

    # =========================================================================
    # Model loading / generation
    # =========================================================================

    @staticmethod
    def _config_name(config: Optional[Any]) -> str:
        if config is None:
            return "None"
        return getattr(config, "__class__", type("x", (), {})).__name__

    @staticmethod
    def _config_model_type(config: Optional[Any]) -> str:
        if config is None:
            return ""
        return str(getattr(config, "model_type", "") or "")

    @staticmethod
    def _resolve_model_route(config: Optional[Any]) -> Dict[str, Any]:
        """
        Centralized model-class router for PyTorch backend.
        """
        config_name = PatentExtractor._config_name(config)
        model_type = PatentExtractor._config_model_type(config)
        is_mistral3 = (model_type == "mistral3") or (config_name == "Mistral3Config")

        if is_mistral3:
            if Mistral3ForCausalLM is not None:
                return {
                    "model_cls": Mistral3ForCausalLM,
                    "pipeline_task": "text-generation",
                    "use_pipeline": True,
                    "trust_remote_code": False,
                    "route_name": "Mistral3ForCausalLM",
                }
            # Some transformers builds expose only Mistral3ForConditionalGeneration.
            # We run a direct model.generate path in that case (no pipeline task support).
            if Mistral3ForConditionalGeneration is not None:
                return {
                    "model_cls": Mistral3ForConditionalGeneration,
                    "pipeline_task": "direct-generate",
                    "use_pipeline": False,
                    "trust_remote_code": False,
                    "route_name": "Mistral3ForConditionalGeneration",
                }
            raise RuntimeError(
                "mistral3 model detected but no supported class is available in this transformers build. "
                "Expected transformers.Mistral3ForCausalLM or transformers.Mistral3ForConditionalGeneration."
            )

        trust_remote_code = False
        if config is not None and getattr(config, "auto_map", None):
            if "AutoModelForCausalLM" in config.auto_map:
                trust_remote_code = True

        return {
            "model_cls": AutoModelForCausalLM,
            "pipeline_task": "text-generation",
            "use_pipeline": True,
            "trust_remote_code": trust_remote_code,
            "route_name": "AutoModelForCausalLM",
        }

    def _resolve_torch_dtype(self, *, device: str, model_type: str):
        """
        Explicit dtype policy.
        A user-forced torch_dtype overrides auto policy.

        Auto policy:
          - CUDA + mistral3: prefer bf16 if supported, else fp16
          - CUDA others: fp16
          - non-CUDA: fp32
        """
        forced = self.torch_dtype
        if forced == "bf16":
            return torch.bfloat16
        if forced == "fp16":
            return torch.float16
        if forced == "fp32":
            return torch.float32

        if device == "cuda":
            if model_type == "mistral3":
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float16
        return torch.float32

    def _is_mistral31_model(self) -> bool:
        return self.model_name.strip() == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    def _load_tokenizer(self, tok_kwargs: Dict[str, Any]):
        if self._is_mistral31_model():
            kwargs = dict(tok_kwargs)
            kwargs["fix_mistral_regex"] = True
            try:
                return AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            except TypeError:
                # transformers version does not support this kwarg yet
                print("⚠️  AutoTokenizer.from_pretrained does not support fix_mistral_regex; continuing without it")
            except Exception:
                # Fall back to standard tokenizer load path
                pass
        return AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)

    def _load_model(self) -> None:
        """Charge le modèle selon le backend configuré."""
        print(f"🧠 Backend: {self.backend}")
        print(f"📦 Model: {self.model_name}")

        if self.backend == "mlx":
            print("⚙️  Loading via MLX")
            global mlx_lm
            if mlx_lm is None:
                try:
                    import mlx_lm as _mlx_lm
                except Exception as e:
                    raise ImportError("MLX backend requested but mlx-lm could not be imported.") from e
                mlx_lm = _mlx_lm
            self.model, self.tokenizer = mlx_lm.load(self.model_name)
            self.pipe = None
            self.pipeline_task = None
            return

        if self.backend == "pytorch":
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            map_arg = self.device_map

            config = None
            try:
                # trust_remote_code=True here only to inspect config safely for custom models
                config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception:
                config = None

            config_name = self._config_name(config)
            model_type = self._config_model_type(config)
            route = self._resolve_model_route(config)
            model_cls = route["model_cls"]
            trust_remote_code = bool(route["trust_remote_code"])
            route_name = str(route["route_name"])
            pipeline_task = str(route["pipeline_task"])
            use_pipeline = bool(route.get("use_pipeline", True))

            dtype = self._resolve_torch_dtype(device=self.device, model_type=model_type)
            print(f"🚀 Loading on {self.device} ({dtype}) via {route_name}")

            tok_kwargs: Dict[str, Any] = {}
            model_kwargs: Dict[str, Any] = {
                "torch_dtype": dtype,
                "device_map": map_arg,
            }
            if trust_remote_code:
                tok_kwargs["trust_remote_code"] = True
                model_kwargs["trust_remote_code"] = True

            try:
                self.tokenizer = self._load_tokenizer(tok_kwargs)
                self.model = model_cls.from_pretrained(self.model_name, **model_kwargs)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load extraction model.\n"
                    f"model_name={self.model_name}\n"
                    f"config_class={config_name}\n"
                    f"model_type={model_type!r}\n"
                    f"route_class={route_name}\n"
                    f"trust_remote_code={trust_remote_code}\n"
                    f"original={e.__class__.__name__}: {e}"
                ) from e

            if use_pipeline:
                # Import lazily: in some environments, importing transformers.pipeline
                # eagerly triggers optional vision deps (torchvision) that may be absent/mismatched.
                from transformers import pipeline

                self.pipe = pipeline(
                    pipeline_task,
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
            else:
                self.pipe = None
            self.pipeline_task = pipeline_task
            return

        raise ValueError(f"Unknown backend: {self.backend}")

    def _generate(self, prompt: str) -> str:
        """Génère du texte avec le modèle chargé."""
        if self.backend == "mlx":
            # mlx_lm.generate signature can vary; keep it simple/reliable.
            output = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=self.max_new_tokens,
            )
            return (output or "").strip()

        if self.backend == "pytorch":
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                # Avoid returning the full prompt+completion string when supported.
                "return_full_text": False,
            }
            if self.do_sample:
                gen_kwargs["temperature"] = self.temperature
            if self.pipeline_task == "text2text-generation":
                gen_kwargs.pop("return_full_text", None)
            if os.getenv("PATENT_EXTRACTOR_DEBUG_GEN", "").strip() == "1" and not self._debug_gen_logged:
                print(
                    "[debug-gen] "
                    f"model_class={self.model.__class__.__name__} "
                    f"pipeline_task={self.pipeline_task} "
                    f"gen_kwargs={gen_kwargs}"
                )
                self._debug_gen_logged = True
            if self.pipe is None or self.pipeline_task == "direct-generate":
                gen_kwargs.pop("return_full_text", None)
                model_inputs = self.tokenizer(prompt, return_tensors="pt")
                if hasattr(self.model, "device"):
                    try:
                        model_inputs = {
                            k: (v.to(self.model.device) if hasattr(v, "to") else v)
                            for k, v in model_inputs.items()
                        }
                    except Exception:
                        pass
                generated = self.model.generate(**model_inputs, **gen_kwargs)
                input_ids = model_inputs.get("input_ids")
                if input_ids is not None and hasattr(input_ids, "shape") and generated.shape[-1] >= input_ids.shape[-1]:
                    new_tokens = generated[0][input_ids.shape[-1] :]
                    return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            try:
                out = self.pipe(prompt, **gen_kwargs)[0]
            except TypeError:
                # Backward compatibility for pipeline versions that do not accept return_full_text.
                gen_kwargs.pop("return_full_text", None)
                out = self.pipe(prompt, **gen_kwargs)[0]
            return out.get("generated_text") or out.get("text") or ""

        raise ValueError(f"Unknown backend: {self.backend}")

    # =========================================================================
    # Prompt helpers
    # =========================================================================

    def set_prompt_template(self, template: str) -> None:
        """
        Permet de changer le template de prompt “à la main”.
        IMPORTANT: recalcule prompt_hash (sinon bug silencieux).
        """
        if "{text}" not in template:
            raise ValueError("Le template doit contenir le placeholder {text}")
        self.prompt_id = None
        self.prompt_template = template
        self.prompt_template_source = "inline_template"
        self.prompt_hash = _sha256(self.prompt_template + self.prompt_suffix)

    def _truncate_ocr(self, text: str) -> str:
        """Tronque le texte OCR si trop long (évite prompts gigantesques)."""
        if len(text) > self.max_ocr_chars:
            return text[: self.max_ocr_chars] + "\n[...] (truncated)"
        return text

    def _should_use_chunked(self, ocr_text: str) -> bool:
        if self.extraction_mode == "chunked":
            return True
        if self.extraction_mode == "single":
            return False
        # auto: only switch when text would be truncated in single mode
        return len(ocr_text) > self.max_ocr_chars

    def _split_text_chunks(self, text: str, offset: int = 0) -> List[str]:
        if not text:
            return []

        step = self.chunk_size_chars - self.chunk_overlap_chars
        start = max(0, int(offset))
        chunks: List[str] = []

        while start < len(text):
            end = min(start + self.chunk_size_chars, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        return chunks

    def _chunk_offsets(self) -> List[int]:
        if self.extraction_passes <= 1:
            return [0]
        # Shift chunk boundaries across passes to recover entities near chunk cuts.
        step = max(1, self.chunk_size_chars // self.extraction_passes)
        offsets = [0]
        for i in range(1, self.extraction_passes):
            offsets.append(i * step)
        return sorted(set(offsets))

    def _first_non_empty(self, values: List[Any]) -> Optional[str]:
        for v in values:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _longest_non_empty(self, values: List[Any]) -> Optional[str]:
        candidates = [v.strip() for v in values if isinstance(v, str) and v.strip()]
        if not candidates:
            return None
        return max(candidates, key=len)

    def _to_date(self, value: Any) -> Optional[date]:
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

    def _merge_entity_lists(self, values: List[Any]) -> Optional[List[Dict[str, Optional[str]]]]:
        seen = set()
        out: List[Dict[str, Optional[str]]] = []

        for value in values:
            normalized = self._normalize_entity_list(value)
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

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, list):
            return len(value) == 0
        return False

    def _normalize_for_vote(self, value: Any) -> str:
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value.strip().lower())
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _choose_scalar(
        self,
        values: List[Any],
        policy: Literal["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"],
        fallback_policy: Literal["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"] = "prefer_non_null",
    ) -> Any:
        if not values:
            return None

        if policy == "prefer_first":
            return values[0]
        if policy == "prefer_last":
            return values[-1]
        if policy == "prefer_non_null":
            for v in values:
                if not self._is_missing(v):
                    return v
            return None

        # vote_majority
        non_missing = [v for v in values if not self._is_missing(v)]
        if non_missing:
            counts = Counter(self._normalize_for_vote(v) for v in non_missing)
            winner_key, _ = counts.most_common(1)[0]
            for v in non_missing:
                if self._normalize_for_vote(v) == winner_key:
                    return v
        if fallback_policy != "vote_majority":
            return self._choose_scalar(values, fallback_policy, fallback_policy="prefer_non_null")
        return None

    def _enforce_date_order(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pub_date = self._to_date(data.get("pub_date_publication"))
        app_date = self._to_date(data.get("pub_date_application"))
        foreign_date = self._to_date(data.get("pub_date_foreign"))

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

    def _merge_metadata_candidates(
        self,
        candidates: List[PatentMetadata],
        *,
        policy: Optional[Literal["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"]] = None,
        fallback_policy: Literal["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"] = "prefer_non_null",
    ) -> PatentMetadata:
        if not candidates:
            return PatentMetadata(identifier="unknown")
        if len(candidates) == 1:
            return candidates[0]

        if policy is None:
            rows = [c.model_dump(mode="json") for c in candidates]
            pub_dates = sorted(d for d in (self._to_date(r.get("pub_date_publication")) for r in rows) if d is not None)
            app_dates = sorted(d for d in (self._to_date(r.get("pub_date_application")) for r in rows) if d is not None)
            foreign_dates = sorted(d for d in (self._to_date(r.get("pub_date_foreign")) for r in rows) if d is not None)

            pub_date = pub_dates[0] if pub_dates else None
            if app_dates:
                app_valid = [d for d in app_dates if pub_date is None or d <= pub_date]
                app_date = app_valid[0] if app_valid else app_dates[0]
            else:
                app_date = None

            if foreign_dates:
                foreign_valid = [d for d in foreign_dates if app_date is None or d <= app_date]
                foreign_date = foreign_valid[0] if foreign_valid else foreign_dates[0]
            else:
                foreign_date = None

            merged_legacy: Dict[str, Any] = {
                "title": self._longest_non_empty([r.get("title") for r in rows]),
                "inventors": self._merge_entity_lists([r.get("inventors") for r in rows]),
                "assignees": self._merge_entity_lists([r.get("assignees") for r in rows]),
                "pub_date_application": app_date.isoformat() if app_date else None,
                "pub_date_publication": pub_date.isoformat() if pub_date else None,
                "pub_date_foreign": foreign_date.isoformat() if foreign_date else None,
                "classification": self._first_non_empty([r.get("classification") for r in rows]),
                "industrial_field": self._first_non_empty([r.get("industrial_field") for r in rows]),
            }
            return self._parse_and_validate(json.dumps(merged_legacy, ensure_ascii=False))

        rows = [c.model_dump(mode="json") for c in candidates]
        merged: Dict[str, Any] = {
            "title": self._choose_scalar([r.get("title") for r in rows], policy, fallback_policy),
            "classification": self._choose_scalar([r.get("classification") for r in rows], policy, fallback_policy),
            "industrial_field": self._choose_scalar([r.get("industrial_field") for r in rows], policy, fallback_policy),
            "pub_date_application": self._choose_scalar(
                [r.get("pub_date_application") for r in rows], policy, fallback_policy
            ),
            "pub_date_publication": self._choose_scalar(
                [r.get("pub_date_publication") for r in rows], policy, fallback_policy
            ),
            "pub_date_foreign": self._choose_scalar([r.get("pub_date_foreign") for r in rows], policy, fallback_policy),
            "inventors": self._merge_entity_lists([r.get("inventors") for r in rows]),
            "assignees": self._merge_entity_lists([r.get("assignees") for r in rows]),
        }
        merged = self._enforce_date_order(merged)
        return self._parse_and_validate(json.dumps(merged, ensure_ascii=False))

    def _missing_critical_fields(self, metadata: PatentMetadata) -> List[str]:
        data = metadata.model_dump(mode="json")
        missing: List[str] = []
        for key in ("title", "inventors", "assignees"):
            if self._is_missing(data.get(key)):
                missing.append(key)
        if self._is_missing(data.get("pub_date_application")):
            missing.append("pub_date_application")
        if self._is_missing(data.get("pub_date_publication")):
            missing.append("pub_date_publication")
        return missing

    def _date_coherence_subscore(self, data: Dict[str, Any]) -> float:
        app_d = self._to_date(data.get("pub_date_application"))
        pub_d = self._to_date(data.get("pub_date_publication"))
        foreign_d = self._to_date(data.get("pub_date_foreign"))
        checks = 0
        valid = 0
        if app_d and pub_d:
            checks += 1
            valid += int(app_d <= pub_d)
        if foreign_d and app_d:
            checks += 1
            valid += int(foreign_d <= app_d)
        if checks == 0:
            return 0.6
        return valid / checks

    def _entity_subscore(self, entities: Any) -> float:
        normalized = self._normalize_entity_list(entities)
        if not normalized:
            return 0.0
        quality = 0.0
        for item in normalized:
            name_ok = bool(str(item.get("name") or "").strip())
            addr_ok = bool(str(item.get("address") or "").strip())
            quality += 0.7 if name_ok else 0.0
            quality += 0.3 if addr_ok else 0.0
        return min(1.0, quality / max(1, len(normalized)))

    def compute_confidence(self, prediction: PatentMetadata, raw_text: str) -> Tuple[float, Dict[str, float]]:
        data = prediction.model_dump(mode="json")
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
        non_null_count = sum(0 if self._is_missing(data.get(k)) else 1 for k in fields)
        completeness = non_null_count / len(fields)

        date_coherence = self._date_coherence_subscore(data)
        inventors_score = self._entity_subscore(data.get("inventors"))
        assignees_score = self._entity_subscore(data.get("assignees"))
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

    # =========================================================================
    # Parsing / normalization
    # =========================================================================

    def _extract_json(self, text: str) -> str:
        """
        Extrait le dernier bloc JSON valide du texte généré.

        Stratégie:
        1) trouver tous les blocs {...} équilibrés (regex récursive)
        2) garder le dernier bloc JSON parsable
        3) fallback sur une extraction partielle à partir de "identifier"
        4) sinon {} (échec)
        """
        candidates = [m.group(0) for m in re.finditer(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)]
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

    def _normalize_entity_list(self, value):
        """Normalise inventors/assignees."""
        if value is None:
            return None

        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return value

        if isinstance(value, str):
            entities = []
            for chunk in value.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                m = re.match(r"(.+?)\s*\(([^)]+)\)", chunk)
                if m:
                    entities.append({"name": m.group(1).strip(), "address": m.group(2).strip()})
                else:
                    entities.append({"name": chunk, "address": None})
            return entities if entities else None

        return None

    def _is_company_name(self, name: str) -> bool:
        """Heuristique: détecte si un nom ressemble à une entreprise."""
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
        for pat in company_patterns:
            if re.search(pat, name_lower):
                return True

        letters = [c for c in name if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.6:
                return True

        return False

    def _fix_inventor_assignee_confusion(self, data: dict) -> dict:
        """Si des entreprises apparaissent dans inventors, les déplacer vers assignees."""
        inventors = data.get("inventors") or []
        assignees = data.get("assignees") or []

        if not inventors:
            return data

        true_inventors = []
        misplaced_companies = []

        for inv in inventors:
            if isinstance(inv, dict):
                name = inv.get("name", "")
                if self._is_company_name(name):
                    misplaced_companies.append(inv)
                else:
                    true_inventors.append(inv)

        if misplaced_companies:
            print(f"🔧 Correction : {len(misplaced_companies)} entreprise(s) déplacée(s) vers assignees")
            data["inventors"] = true_inventors if true_inventors else None
            data["assignees"] = (assignees + misplaced_companies) or None

        return data

    def _fix_duplicate_dates(self, data: dict) -> dict:
        """Corrige des dates dupliquées (heuristique pragmatique)."""
        app_date = data.get("pub_date_application")
        pub_date = data.get("pub_date_publication")
        foreign_date = data.get("pub_date_foreign")

        if app_date and pub_date and app_date == pub_date:
            data["pub_date_application"] = None

        if app_date and pub_date and foreign_date and app_date == pub_date == foreign_date:
            data["pub_date_application"] = None
            data["pub_date_foreign"] = None

        return data

    def _parse_and_validate(self, json_str: str) -> PatentMetadata:
        """Parse le JSON et valide avec Pydantic."""
        try:
            data = json.loads(json_str)

            if not isinstance(data, dict):
                print(f"⚠️ JSON type inattendu: {type(data)}")
                data = data[0] if isinstance(data, list) and data else {}

            # rétrocompatibilité (noms de champs)
            if "assignee" in data and "assignees" not in data:
                data["assignees"] = data.pop("assignee")
            if "inventor" in data and "inventors" not in data:
                data["inventors"] = data.pop("inventor")
            if "class" in data and "classification" not in data:
                data["classification"] = data.pop("class")

            # champs requis (même si null)
            required_fields = [
                "title",
                "inventors",
                "assignees",
                "pub_date_application",
                "pub_date_publication",
                "pub_date_foreign",
                "classification",
                "industrial_field",
            ]
            for k in required_fields:
                data.setdefault(k, None)

            # normalisation inventors/assignees
            data["inventors"] = self._normalize_entity_list(data.get("inventors"))
            data["assignees"] = self._normalize_entity_list(data.get("assignees"))

            # corrections heuristiques
            data = self._fix_inventor_assignee_confusion(data)
            data = self._fix_duplicate_dates(data)

            return PatentMetadata(**data)

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            print(f"⚠️  Erreur de validation JSON: {e}")
            print(f"→ JSON brut:\n{json_str}\n")
            return PatentMetadata(identifier="unknown")

    # =========================================================================
    # Extraction (with timings)
    # =========================================================================

    def _extract_single_metadata(
        self,
        ocr_text: str,
        *,
        debug: bool = False,
        truncate: bool = True,
        prompt_suffix_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        do_sample_override: Optional[bool] = None,
    ) -> Tuple[PatentMetadata, Optional[Dict[str, float]], Optional[str]]:
        t0 = time.perf_counter() if self.timings != "off" else None
        text_for_prompt = self._truncate_ocr(ocr_text) if truncate else ocr_text

        t_prompt0 = time.perf_counter() if self.timings == "detailed" else None
        suffix = prompt_suffix_override if prompt_suffix_override is not None else self.prompt_suffix
        prompt = self.prompt_template.format(text=text_for_prompt) + suffix
        t_prompt1 = time.perf_counter() if self.timings == "detailed" else None

        if debug:
            print("=" * 80)
            print("📝 PROMPT ENVOYÉ AU MODÈLE:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        t_gen0 = time.perf_counter() if self.timings != "off" else None
        original_temperature = self.temperature
        original_do_sample = self.do_sample
        if temperature_override is not None:
            self.temperature = temperature_override
        if do_sample_override is not None:
            self.do_sample = do_sample_override
        try:
            raw_output = self._generate(prompt)
        finally:
            self.temperature = original_temperature
            self.do_sample = original_do_sample
        t_gen1 = time.perf_counter() if self.timings != "off" else None

        if debug:
            print("\n" + "=" * 80)
            print("🤖 SORTIE BRUTE DU MODÈLE:")
            print("=" * 80)
            print(raw_output)
            print("=" * 80)

        t_parse0 = time.perf_counter() if self.timings == "detailed" else None
        json_str = self._extract_json(raw_output)

        if debug:
            print("\n" + "=" * 80)
            print("📦 JSON EXTRAIT:")
            print("=" * 80)
            print(json_str)
            print("=" * 80 + "\n")

        metadata = self._parse_and_validate(json_str)
        t_parse1 = time.perf_counter() if self.timings == "detailed" else None
        t_end = time.perf_counter() if self.timings != "off" else None

        timing = self._timing_dict(
            t0=t0,
            t_prompt0=t_prompt0,
            t_prompt1=t_prompt1,
            t_gen0=t_gen0,
            t_gen1=t_gen1,
            t_parse0=t_parse0,
            t_parse1=t_parse1,
            t_end=t_end,
        )
        raw_out = raw_output if self.save_raw_output else None
        return metadata, timing, raw_out

    def _extract_chunked_metadata(
        self,
        ocr_text: str,
        *,
        debug: bool = False,
        merge_policy: Optional[Literal["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"]] = None,
        prompt_suffix_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        do_sample_override: Optional[bool] = None,
    ) -> Tuple[PatentMetadata, Optional[Dict[str, float]], Optional[List[str]]]:
        t0 = time.perf_counter() if self.timings != "off" else None

        candidates: List[PatentMetadata] = []
        raw_outputs: List[str] = []
        sum_generate = 0.0
        sum_prompt = 0.0
        sum_parse = 0.0
        chunk_count = 0
        pass_count = 0

        for offset in self._chunk_offsets():
            chunks = self._split_text_chunks(ocr_text, offset=offset)
            if not chunks:
                continue
            pass_count += 1
            for chunk in chunks:
                chunk_count += 1
                metadata, timing, raw_output = self._extract_single_metadata(
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
            return self._extract_single_metadata(ocr_text, debug=debug, truncate=True)

        t_merge0 = time.perf_counter() if self.timings != "off" else None
        merged = self._merge_metadata_candidates(candidates, policy=merge_policy)
        t_merge1 = time.perf_counter() if self.timings != "off" else None

        if self.timings == "off":
            return merged, None, (raw_outputs if self.save_raw_output else None)

        t_end = time.perf_counter()
        timing: Dict[str, float] = {
            "t_total_s": max(0.0, t_end - (t0 or t_end)),
            "t_generate_s": max(0.0, sum_generate),
            "n_chunks": float(chunk_count),
            "n_passes": float(pass_count),
        }
        if self.timings == "detailed":
            timing["t_prompt_s"] = max(0.0, sum_prompt)
            timing["t_parse_s"] = max(0.0, sum_parse)
        if t_merge0 is not None and t_merge1 is not None:
            timing["t_chunk_merge_s"] = max(0.0, t_merge1 - t_merge0)

        return merged, timing, (raw_outputs if self.save_raw_output else None)

    def _run_baseline_strategy(
        self,
        ocr_text: str,
        *,
        debug: bool = False,
        prompt_suffix_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        do_sample_override: Optional[bool] = None,
    ) -> Tuple[PatentMetadata, Optional[Dict[str, float]], Any, Dict[str, Any]]:
        if self._should_use_chunked(ocr_text):
            metadata, timing, raw_output = self._extract_chunked_metadata(
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
            metadata, timing, raw_output = self._extract_single_metadata(
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

    def _run_chunked_strategy(self, ocr_text: str, *, debug: bool = False) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
        metadata, timing, raw_output = self._extract_chunked_metadata(
            ocr_text,
            debug=debug,
            merge_policy=self.merge_policy,
        )
        chunks_count = int((timing or {}).get("n_chunks", 0))
        model_calls = chunks_count if chunks_count > 0 else 1
        meta = {
            "strategy_used": "chunked",
            "was_rerun": False,
            "pass_count": model_calls,
            "chunks_count": chunks_count,
            "header_first_used": False,
            "merge_policy_used": self.merge_policy,
            "self_consistency_n_used": 1,
            "timing": timing or {},
        }
        return metadata, meta, raw_output

    def _run_header_first_strategy(self, ocr_text: str, *, debug: bool = False) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
        lines = ocr_text.splitlines()
        header_text = "\n".join(lines[: self.header_lines]) if lines else ocr_text

        header_meta, timing_header, raw_header = self._extract_single_metadata(header_text, debug=debug, truncate=False)
        missing = self._missing_critical_fields(header_meta)
        fallback_to_full = bool(missing)

        if fallback_to_full:
            full_meta, timing_full, raw_full, base_meta = self._run_baseline_strategy(ocr_text, debug=debug)
            merged = self._merge_metadata_candidates([header_meta, full_meta], policy=self.merge_policy)
            timing: Dict[str, float] = {}
            if timing_header:
                timing.update({f"header_{k}": v for k, v in timing_header.items()})
            if timing_full:
                timing.update({f"pass1_{k}": v for k, v in timing_full.items()})
            raw_output = {"header": raw_header, "full_text": raw_full}
            pass_count = 1 + int(base_meta.get("pass_count", 1))
            chunks_count = 1 + int(base_meta.get("chunks_count", 1))
        else:
            merged = header_meta
            timing = {f"header_{k}": v for k, v in (timing_header or {}).items()}
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
            "merge_policy_used": self.merge_policy,
            "self_consistency_n_used": 1,
            "timing": timing,
        }
        return merged, meta, raw_output

    def _run_two_pass_targeted_strategy(
        self, ocr_text: str, *, debug: bool = False
    ) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
        pass1_meta, timing1, raw1, pass1_info = self._run_baseline_strategy(ocr_text, debug=debug)
        conf1, conf1_sub = self.compute_confidence(pass1_meta, ocr_text)

        was_rerun = conf1 < self.targeted_rerun_threshold
        correction_suffix = (
            "\n\nCorrection mode: Re-check missing/uncertain fields and date consistency against the text. "
            "Output only corrected JSON.\n"
        )

        if was_rerun:
            pass2_meta, timing2, raw2, pass2_info = self._run_baseline_strategy(
                ocr_text,
                debug=debug,
                prompt_suffix_override=self.prompt_suffix + correction_suffix,
            )
            merged = self._merge_metadata_candidates([pass1_meta, pass2_meta], policy=self.merge_policy)
            timing: Dict[str, float] = {}
            if timing1:
                timing.update({f"pass1_{k}": v for k, v in timing1.items()})
            if timing2:
                timing.update({f"pass2_{k}": v for k, v in timing2.items()})
            pass_count = int(pass1_info.get("pass_count", 1)) + int(pass2_info.get("pass_count", 1))
            chunks_count = int(pass1_info.get("chunks_count", 1)) + int(pass2_info.get("chunks_count", 1))
            raw_output = {"pass1": raw1, "pass2": raw2}
        else:
            merged = pass1_meta
            timing = {f"pass1_{k}": v for k, v in (timing1 or {}).items()}
            pass_count = int(pass1_info.get("pass_count", 1))
            chunks_count = int(pass1_info.get("chunks_count", 1))
            raw_output = {"pass1": raw1}

        meta = {
            "strategy_used": "two_pass_targeted",
            "was_rerun": was_rerun,
            "targeted_rerun_threshold": self.targeted_rerun_threshold,
            "pass1_confidence": conf1,
            "pass1_confidence_subscores": conf1_sub,
            "pass_count": pass_count,
            "chunks_count": chunks_count,
            "header_first_used": False,
            "merge_policy_used": self.merge_policy,
            "self_consistency_n_used": 1,
            "timing": timing,
        }
        return merged, meta, raw_output

    def _field_variance(self, rows: List[Dict[str, Any]], field: str) -> float:
        values = [r.get(field) for r in rows if not self._is_missing(r.get(field))]
        if len(values) <= 1:
            return 0.0
        uniq = len({self._normalize_for_vote(v) for v in values})
        return max(0.0, min(1.0, (uniq - 1) / (len(values) - 1)))

    def _run_self_consistency_strategy(
        self, ocr_text: str, *, debug: bool = False
    ) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
        n = max(1, self.self_consistency_n)
        candidates: List[PatentMetadata] = []
        all_timings: List[Dict[str, float]] = []
        all_raw: List[Any] = []
        pass_count = 0
        chunks_count = 0

        t_merge0 = time.perf_counter() if self.timings != "off" else None
        for _ in range(n):
            pred, timing, raw_out, info = self._run_baseline_strategy(
                ocr_text,
                debug=debug,
                temperature_override=self.self_consistency_temp,
                do_sample_override=(self.self_consistency_temp > 0.0 or self.do_sample),
            )
            candidates.append(pred)
            pass_count += int(info.get("pass_count", 1))
            chunks_count += int(info.get("chunks_count", 1))
            if timing:
                all_timings.append(timing)
            all_raw.append(raw_out)

        merged = self._merge_metadata_candidates(
            candidates,
            policy="vote_majority",
            fallback_policy="prefer_non_null",
        )
        t_merge1 = time.perf_counter() if self.timings != "off" else None

        agg_timing: Dict[str, float] = {}
        if all_timings:
            for t in all_timings:
                for k, v in t.items():
                    agg_timing[k] = agg_timing.get(k, 0.0) + float(v)
        if t_merge0 is not None and t_merge1 is not None:
            agg_timing["t_self_consistency_merge_s"] = max(0.0, t_merge1 - t_merge0)

        rows = [c.model_dump(mode="json") for c in candidates]
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
        variance_by_field = {f: round(self._field_variance(rows, f), 6) for f in fields}
        variance_mean = round(sum(variance_by_field.values()) / len(variance_by_field), 6)

        meta = {
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
            "timing": agg_timing,
        }
        return merged, meta, all_raw

    def _run_strategy(self, ocr_text: str, *, debug: bool = False) -> Tuple[PatentMetadata, Dict[str, Any], Any]:
        if self.strategy == "baseline":
            pred, timing, raw_output, base_meta = self._run_baseline_strategy(ocr_text, debug=debug)
            base_meta["strategy_used"] = "baseline"
            base_meta["timing"] = timing or {}
            return pred, base_meta, raw_output
        if self.strategy == "chunked":
            return self._run_chunked_strategy(ocr_text, debug=debug)
        if self.strategy == "header_first":
            return self._run_header_first_strategy(ocr_text, debug=debug)
        if self.strategy == "two_pass_targeted":
            return self._run_two_pass_targeted_strategy(ocr_text, debug=debug)
        if self.strategy == "self_consistency":
            return self._run_self_consistency_strategy(ocr_text, debug=debug)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def extract(self, ocr_text: str, debug: bool = False) -> PatentExtraction:
        """
        Extrait les métadonnées structurées d'un texte OCR.

        Timings:
        - basic: t_generate_s, t_total_s
        - detailed: + t_prompt_s, t_parse_s
        """
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

    def _timing_dict(
        self,
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
        if self.timings == "off" or t0 is None or t_end is None:
            return None

        out: Dict[str, float] = {}
        out["t_total_s"] = max(0.0, t_end - t0)

        if t_gen0 is not None and t_gen1 is not None:
            out["t_generate_s"] = max(0.0, t_gen1 - t_gen0)

        if self.timings == "detailed":
            if t_prompt0 is not None and t_prompt1 is not None:
                out["t_prompt_s"] = max(0.0, t_prompt1 - t_prompt0)
            if t_parse0 is not None and t_parse1 is not None:
                out["t_parse_s"] = max(0.0, t_parse1 - t_parse0)

        return out

    # =========================================================================
    # File-level wrapper (JSONL record)
    # =========================================================================

    def extract_from_file(self, txt_path: Path, raw_output_dir: Optional[Path] = None) -> dict:
        """
        Extrait les métadonnées d'un fichier .txt.

        Returns:
            Dict sérialisable en JSON (prêt pour JSONL).
            Ajoute prompt_id/prompt_hash et timing si dispo.
        """
        t_file0 = time.perf_counter() if self.timings != "off" else None

        try:
            # Lecture OCR
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

            # Extraction LLM
            extraction = self.extract(ocr_text)
            record = extraction.model_dump(mode="json")

            # Champs utiles pour le bench
            record["file_name"] = txt_path.name
            record["ocr_path"] = str(txt_path)

            # Identifier depuis le nom de fichier
            if isinstance(record.get("prediction"), dict):
                record["prediction"]["identifier"] = txt_path.stem.split("_")[0]

            # Dimensions de benchmark
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

            # Timings
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
                    # Avoid bloating JSONL when a file is persisted.
                    record.pop("raw_output", None)

            return record

        except Exception as e:
            print(f"⚠️ Erreur sur {txt_path.name}: {e}")
            traceback.print_exc()

            rec = {
                "file_name": txt_path.name,
                "ocr_path": str(txt_path),
                "error": f"exception: {e.__class__.__name__}",
                "strategy_used": self.strategy,
                "confidence_score": 0.0,
            }
            if self.prompt_id is not None:
                rec["prompt_id"] = self.prompt_id
            rec["prompt_hash"] = self.prompt_hash

            if self.timings != "off" and t_file0 is not None:
                rec["timing"] = {"t_total_file_s": max(0.0, time.perf_counter() - t_file0)}

            return rec

    # =========================================================================
    # Batch runner
    # =========================================================================

    def batch_extract(
        self,
        txt_dir: Path,
        out_file: Path,
        limit: Optional[int] = None,
        raw_output_dir: Optional[Path] = None,
    ) -> int:
        """
        Traite un dossier de fichiers .txt en batch.

        Returns:
            nombre de documents traités
        """
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
