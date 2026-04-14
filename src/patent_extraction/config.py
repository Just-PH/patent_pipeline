from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from .guardrails import GUARDRAIL_PROFILES
from .strategies import DEFAULT_STRATEGY, STRATEGY_NAMES


TorchDType = Literal["auto", "bf16", "fp16", "fp32"]
TokenizerMode = Literal["auto", "mistral"]
TimingsMode = Literal["off", "basic", "detailed"]
ExtractionMode = Literal["single", "chunked", "auto"]
MergePolicy = Literal["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"]


@dataclass(frozen=True)
class VLLMConfig:
    enable_prefix_caching: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    swap_space: float = 4.0
    enforce_eager: bool = False
    doc_batch_size: Optional[int] = 32
    sort_by_prompt_length: bool = True
    tokenizer_mode: TokenizerMode = "auto"
    quantization: str = "none"

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_model_len is not None and self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        if self.swap_space < 0.0:
            raise ValueError("swap_space must be >= 0")
        if self.doc_batch_size is not None and self.doc_batch_size < 1:
            raise ValueError("doc_batch_size must be >= 1")
        if self.tokenizer_mode not in {"auto", "mistral"}:
            raise ValueError("tokenizer_mode must be one of: auto|mistral")


@dataclass(frozen=True)
class StrategyConfig:
    name: str = DEFAULT_STRATEGY
    extraction_mode: ExtractionMode = "auto"
    chunk_size_chars: int = 7000
    chunk_overlap_chars: int = 800
    extraction_passes: int = 2
    header_lines: int = 30
    targeted_rerun_threshold: float = 0.6
    self_consistency_n: int = 2
    self_consistency_temp: float = 0.2
    merge_policy: MergePolicy = "prefer_non_null"

    def __post_init__(self) -> None:
        if self.name not in STRATEGY_NAMES:
            raise ValueError(f"strategy.name must be one of: {list(STRATEGY_NAMES)}")
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


@dataclass(frozen=True)
class ExtractionConfig:
    model_name: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    backend: Literal["vllm"] = "vllm"
    torch_dtype: TorchDType = "auto"
    prompt_id: Optional[str] = None
    prompt_path: Optional[Path] = None
    guardrail_profile: str = "auto"
    max_ocr_chars: int = 10000
    max_new_tokens: int = 1024
    temperature: float = 0.0
    do_sample: bool = False
    save_strategy_meta: bool = False
    save_raw_output: bool = False
    timings: TimingsMode = "basic"
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    def __post_init__(self) -> None:
        if self.max_ocr_chars < 1:
            raise ValueError("max_ocr_chars must be >= 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.guardrail_profile not in GUARDRAIL_PROFILES:
            raise ValueError(f"guardrail_profile must be one of: {sorted(GUARDRAIL_PROFILES)}")

    def with_overrides(self, **overrides: Any) -> "ExtractionConfig":
        vllm_cfg = self.vllm
        strategy_cfg = self.strategy
        root_updates: Dict[str, Any] = {}

        for key, value in overrides.items():
            if value is None:
                continue
            if key == "strategy":
                strategy_cfg = replace(strategy_cfg, name=value)
                continue
            if hasattr(vllm_cfg, key):
                vllm_cfg = replace(vllm_cfg, **{key: value})
                continue
            if hasattr(strategy_cfg, key):
                strategy_cfg = replace(strategy_cfg, **{key: value})
                continue
            if key == "prompt_path":
                value = Path(value)
            root_updates[key] = value

        return replace(self, vllm=vllm_cfg, strategy=strategy_cfg, **root_updates)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.prompt_path is not None:
            data["prompt_path"] = str(self.prompt_path)
        return data


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    description: str = ""
    definition_path: Optional[Path] = None
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    def read_prompt_text(self) -> Optional[str]:
        if self.extraction.prompt_path is None:
            return None
        return self.extraction.prompt_path.read_text(encoding="utf-8")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "definition_path": str(self.definition_path) if self.definition_path is not None else None,
            "extraction": self.extraction.as_dict(),
        }


__all__ = ["ExtractionConfig", "ProfileConfig", "StrategyConfig", "TorchDType", "VLLMConfig"]
