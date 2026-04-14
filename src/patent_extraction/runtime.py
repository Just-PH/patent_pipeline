from __future__ import annotations

import importlib.util
from typing import Any, Dict, Optional


vllm_mod = None
HAS_BITSANDBYTES = importlib.util.find_spec("bitsandbytes") is not None


def require_vllm():
    global vllm_mod
    if vllm_mod is None:
        try:
            import vllm as _vllm
        except Exception as exc:
            raise ImportError("vLLM backend requested but vllm could not be imported.") from exc
        vllm_mod = _vllm
    return vllm_mod


def resolve_vllm_dtype(torch_dtype: str) -> str:
    forced = str(torch_dtype or "auto").strip().lower()
    if forced == "bf16":
        return "bfloat16"
    if forced == "fp16":
        return "float16"
    if forced == "fp32":
        return "float32"
    return "auto"


def resolve_vllm_quantization(quantization: Optional[str]) -> Optional[str]:
    value = str(quantization or "none").strip().lower()
    if value in {"", "none"}:
        return None
    aliases = {
        "bnb_4bit": "bitsandbytes",
        "bnb_8bit": "bitsandbytes",
        "4bit": "bitsandbytes",
        "4-bit": "bitsandbytes",
        "bitsandbytes_4bit": "bitsandbytes",
        "bitsandbytes_8bit": "bitsandbytes",
    }
    resolved = aliases.get(value, value)
    if resolved == "bitsandbytes" and not HAS_BITSANDBYTES:
        raise ImportError("quantization=bitsandbytes requested but bitsandbytes is not installed in the image.")
    return resolved


def build_sampling_params(
    owner: Any,
    *,
    temperature_override: Optional[float] = None,
    do_sample_override: Optional[bool] = None,
):
    vllm = require_vllm()
    do_sample = owner.do_sample if do_sample_override is None else do_sample_override
    temperature = owner.temperature if temperature_override is None else temperature_override
    return vllm.SamplingParams(
        max_tokens=owner.max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        skip_special_tokens=True,
    )


def build_llm_kwargs(owner: Any) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = {
        "model": owner.model_name,
        "tensor_parallel_size": owner.tensor_parallel_size,
        "dtype": resolve_vllm_dtype(owner.torch_dtype),
        "gpu_memory_utilization": owner.gpu_memory_utilization,
        "swap_space": owner.swap_space,
        "enable_prefix_caching": owner.enable_prefix_caching,
        "enforce_eager": owner.enforce_eager,
    }
    if owner.max_model_len is not None:
        llm_kwargs["max_model_len"] = owner.max_model_len
    if owner.tokenizer_mode != "auto":
        llm_kwargs["tokenizer_mode"] = owner.tokenizer_mode
    quantization = resolve_vllm_quantization(owner.quantization)
    if quantization is not None:
        llm_kwargs["quantization"] = quantization
    return llm_kwargs


def load_model(owner: Any) -> None:
    vllm = require_vllm()
    llm_kwargs = build_llm_kwargs(owner)
    dtype = llm_kwargs.get("dtype", "auto")
    print(
        "⚙️  Loading via vLLM "
        f"(dtype={dtype}, tp={owner.tensor_parallel_size}, "
        f"prefix_caching={owner.enable_prefix_caching}, "
        f"quantization={llm_kwargs.get('quantization', 'none')})"
    )
    owner.model = vllm.LLM(**llm_kwargs)
    owner.tokenizer = None
    if hasattr(owner.model, "get_tokenizer"):
        try:
            owner.tokenizer = owner.model.get_tokenizer()
        except Exception:
            owner.tokenizer = None


def generate_text(
    owner: Any,
    prompt: str,
    *,
    temperature_override: Optional[float] = None,
    do_sample_override: Optional[bool] = None,
) -> str:
    sampling_params = build_sampling_params(
        owner,
        temperature_override=temperature_override,
        do_sample_override=do_sample_override,
    )
    outputs = owner.model.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    if not outputs:
        return ""
    first = outputs[0]
    candidates = getattr(first, "outputs", None) or []
    if not candidates:
        return ""
    return (getattr(candidates[0], "text", None) or "").strip()


__all__ = [
    "HAS_BITSANDBYTES",
    "build_llm_kwargs",
    "build_sampling_params",
    "generate_text",
    "load_model",
    "require_vllm",
    "resolve_vllm_dtype",
    "resolve_vllm_quantization",
]
