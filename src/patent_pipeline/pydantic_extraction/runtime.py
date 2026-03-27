from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
from typing import Any, Callable, Dict, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore

try:
    from transformers import Mistral3ForCausalLM  # type: ignore
except Exception:
    Mistral3ForCausalLM = None  # type: ignore

try:
    from transformers import Mistral3ForConditionalGeneration  # type: ignore
except Exception:
    Mistral3ForConditionalGeneration = None  # type: ignore


mlx_lm = None
vllm_mod = None
HAS_MLX = importlib.util.find_spec("mlx_lm") is not None


def require_vllm():
    global vllm_mod
    if vllm_mod is None:
        try:
            import vllm as _vllm
        except Exception as e:
            raise ImportError("vLLM backend requested but vllm could not be imported.") from e
        vllm_mod = _vllm
    return vllm_mod


def resolve_attn_implementation(owner: Any, *, dtype: Any) -> Optional[str]:
    attn_implementation = str(getattr(owner, "attn_implementation", "auto") or "auto").strip().lower()
    if attn_implementation == "auto":
        return None
    if attn_implementation == "flash_attention_2":
        if getattr(owner, "device", "") != "cuda":
            raise ValueError("attn_implementation=flash_attention_2 requires a CUDA device")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("attn_implementation=flash_attention_2 requires torch_dtype fp16 or bf16")
    return attn_implementation


def build_quantization_config(owner: Any, *, dtype: Any):
    quantization = str(getattr(owner, "quantization", "none") or "none").strip().lower()
    if quantization == "none":
        return None
    if BitsAndBytesConfig is None:
        raise ImportError(
            "Quantization requested but transformers.BitsAndBytesConfig is unavailable. "
            "Install a transformers build with bitsandbytes support."
        )
    if quantization == "bnb_8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "bnb_4bit":
        kwargs: Dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        }
        if dtype in {torch.float16, torch.bfloat16, torch.float32}:
            kwargs["bnb_4bit_compute_dtype"] = dtype
        return BitsAndBytesConfig(**kwargs)
    raise ValueError(f"Unknown quantization mode: {quantization}")


def resolve_cache_implementation(owner: Any) -> Optional[str]:
    cache_implementation = str(getattr(owner, "cache_implementation", "auto") or "auto").strip().lower()
    if cache_implementation in {"", "auto", "dynamic"}:
        return None
    if cache_implementation in {"static", "offloaded_static"} and not has_c_compiler():
        raise RuntimeError(
            "cache_implementation="
            f"{cache_implementation} requires a visible C compiler on this stack. "
            "Static cache falls into torch.compile/TorchInductor during generate(), "
            "but no compiler was found via CC, cc, gcc, or clang. "
            "Use cache_implementation=dynamic|auto, or install a compiler in the image."
        )
    return cache_implementation


def has_c_compiler() -> bool:
    cc_env = str(os.getenv("CC", "") or "").strip()
    candidates = []
    if cc_env:
        try:
            parts = shlex.split(cc_env)
        except ValueError:
            parts = [cc_env]
        if parts:
            candidates.append(parts[0])
    candidates.extend(["cc", "gcc", "clang"])
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return True
    return False


def build_generate_kwargs(
    owner: Any,
    *,
    temperature_override: Optional[float] = None,
    do_sample_override: Optional[bool] = None,
) -> Dict[str, Any]:
    do_sample = owner.do_sample if do_sample_override is None else do_sample_override
    temperature = owner.temperature if temperature_override is None else temperature_override
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": owner.max_new_tokens,
        "do_sample": do_sample,
        "return_full_text": False,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
    cache_implementation = resolve_cache_implementation(owner)
    if cache_implementation is not None:
        gen_kwargs["cache_implementation"] = cache_implementation
    return gen_kwargs


def resolve_vllm_dtype(owner: Any) -> str:
    forced = str(getattr(owner, "torch_dtype", "auto") or "auto").strip().lower()
    if forced == "bf16":
        return "bfloat16"
    if forced == "fp16":
        return "float16"
    if forced == "fp32":
        return "float32"
    return "auto"


def build_vllm_sampling_params(
    owner: Any,
    *,
    temperature_override: Optional[float] = None,
    do_sample_override: Optional[bool] = None,
):
    vllm = require_vllm()
    do_sample = owner.do_sample if do_sample_override is None else do_sample_override
    temperature = owner.temperature if temperature_override is None else temperature_override
    kwargs: Dict[str, Any] = {
        "max_tokens": owner.max_new_tokens,
        "temperature": temperature if do_sample else 0.0,
        "skip_special_tokens": True,
    }
    return vllm.SamplingParams(**kwargs)


def config_name(config: Optional[Any]) -> str:
    if config is None:
        return "None"
    return getattr(config, "__class__", type("x", (), {})).__name__


def config_model_type(config: Optional[Any]) -> str:
    if config is None:
        return ""
    return str(getattr(config, "model_type", "") or "")


def resolve_model_route(
    config: Optional[Any],
    *,
    auto_model_cls: Any = AutoModelForCausalLM,
    mistral3_causal_cls: Any = Mistral3ForCausalLM,
    mistral3_conditional_cls: Any = Mistral3ForConditionalGeneration,
) -> Dict[str, Any]:
    """
    Centralized model-class router for the PyTorch backend.
    """
    cfg_name = config_name(config)
    model_type = config_model_type(config)
    is_mistral3 = (model_type == "mistral3") or (cfg_name == "Mistral3Config")

    if is_mistral3:
        if mistral3_causal_cls is not None:
            return {
                "model_cls": mistral3_causal_cls,
                "pipeline_task": "text-generation",
                "use_pipeline": True,
                "trust_remote_code": False,
                "route_name": "Mistral3ForCausalLM",
            }
        if mistral3_conditional_cls is not None:
            return {
                "model_cls": mistral3_conditional_cls,
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
        "model_cls": auto_model_cls,
        "pipeline_task": "text-generation",
        "use_pipeline": True,
        "trust_remote_code": trust_remote_code,
        "route_name": "AutoModelForCausalLM",
    }


def resolve_torch_dtype(owner: Any, *, device: str, model_type: str):
    """
    Explicit dtype policy.
    A user-forced torch_dtype overrides auto policy.

    Auto policy:
      - CUDA + mistral3: prefer bf16 if supported, else fp16
      - CUDA others: fp16
      - non-CUDA: fp32
    """
    forced = owner.torch_dtype
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


def is_mistral31_model(owner: Any) -> bool:
    return owner.model_name.strip() == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


def load_tokenizer(owner: Any, tok_kwargs: Dict[str, Any]):
    if owner._is_mistral31_model():
        kwargs = dict(tok_kwargs)
        kwargs["fix_mistral_regex"] = True
        try:
            return AutoTokenizer.from_pretrained(owner.model_name, **kwargs)
        except TypeError:
            print("⚠️  AutoTokenizer.from_pretrained does not support fix_mistral_regex; continuing without it")
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(owner.model_name, **tok_kwargs)


def load_model_into(
    owner: Any,
    *,
    resolve_model_route_fn: Callable[[Optional[Any]], Dict[str, Any]],
    resolve_torch_dtype_fn: Callable[..., Any],
    load_tokenizer_fn: Callable[[Dict[str, Any]], Any],
) -> None:
    print(f"🧠 Backend: {owner.backend}")
    print(f"📦 Model: {owner.model_name}")

    if owner.backend == "mlx":
        print("⚙️  Loading via MLX")
        global mlx_lm
        if mlx_lm is None:
            try:
                import mlx_lm as _mlx_lm
            except Exception as e:
                raise ImportError("MLX backend requested but mlx-lm could not be imported.") from e
            mlx_lm = _mlx_lm
        owner.model, owner.tokenizer = mlx_lm.load(owner.model_name)
        owner.pipe = None
        owner.pipeline_task = None
        return

    if owner.backend == "vllm":
        vllm = require_vllm()
        dtype = resolve_vllm_dtype(owner)
        llm_kwargs: Dict[str, Any] = {
            "model": owner.model_name,
            "tensor_parallel_size": owner.vllm_tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": owner.vllm_gpu_memory_utilization,
            "swap_space": owner.vllm_swap_space,
            "enable_prefix_caching": owner.vllm_enable_prefix_caching,
            "enforce_eager": owner.vllm_enforce_eager,
        }
        if owner.vllm_max_model_len is not None:
            llm_kwargs["max_model_len"] = owner.vllm_max_model_len
        if getattr(owner, "vllm_tokenizer_mode", "auto") != "auto":
            llm_kwargs["tokenizer_mode"] = owner.vllm_tokenizer_mode
        print(
            "⚙️  Loading via vLLM "
            f"(dtype={dtype}, tp={owner.vllm_tensor_parallel_size}, "
            f"prefix_caching={owner.vllm_enable_prefix_caching})"
        )
        owner.model = vllm.LLM(**llm_kwargs)
        owner.tokenizer = None
        if hasattr(owner.model, "get_tokenizer"):
            try:
                owner.tokenizer = owner.model.get_tokenizer()
            except Exception:
                owner.tokenizer = None
        owner.pipe = None
        owner.pipeline_task = "vllm-generate"
        return

    if owner.backend == "pytorch":
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        map_arg = owner.device_map

        config = None
        try:
            config = AutoConfig.from_pretrained(owner.model_name, trust_remote_code=True)
        except Exception:
            config = None

        cfg_name = config_name(config)
        model_type = config_model_type(config)
        route = resolve_model_route_fn(config)
        model_cls = route["model_cls"]
        trust_remote_code = bool(route["trust_remote_code"])
        route_name = str(route["route_name"])
        pipeline_task = str(route["pipeline_task"])
        use_pipeline = bool(route.get("use_pipeline", True))

        dtype = resolve_torch_dtype_fn(device=owner.device, model_type=model_type)
        print(f"🚀 Loading on {owner.device} ({dtype}) via {route_name}")

        tok_kwargs: Dict[str, Any] = {}
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": map_arg,
        }
        attn_implementation = resolve_attn_implementation(owner, dtype=dtype)
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        quantization_config = build_quantization_config(owner, dtype=dtype)
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if trust_remote_code:
            tok_kwargs["trust_remote_code"] = True
            model_kwargs["trust_remote_code"] = True

        try:
            owner.tokenizer = load_tokenizer_fn(tok_kwargs)
            owner.model = model_cls.from_pretrained(owner.model_name, **model_kwargs)
        except Exception as e:
            raise RuntimeError(
                "Failed to load extraction model.\n"
                f"model_name={owner.model_name}\n"
                f"config_class={cfg_name}\n"
                f"model_type={model_type!r}\n"
                f"route_class={route_name}\n"
                f"trust_remote_code={trust_remote_code}\n"
                f"original={e.__class__.__name__}: {e}"
            ) from e

        if use_pipeline:
            from transformers import pipeline

            owner.pipe = pipeline(
                pipeline_task,
                model=owner.model,
                tokenizer=owner.tokenizer,
            )
        else:
            owner.pipe = None
        owner.pipeline_task = pipeline_task
        return

    raise ValueError(f"Unknown backend: {owner.backend}")


def generate_text(
    owner: Any,
    prompt: str,
    *,
    temperature_override: Optional[float] = None,
    do_sample_override: Optional[bool] = None,
) -> str:
    if owner.backend == "mlx":
        output = mlx_lm.generate(
            owner.model,
            owner.tokenizer,
            prompt,
            max_tokens=owner.max_new_tokens,
        )
        return (output or "").strip()

    if owner.backend == "pytorch":
        gen_kwargs = build_generate_kwargs(
            owner,
            temperature_override=temperature_override,
            do_sample_override=do_sample_override,
        )
        if owner.pipeline_task == "text2text-generation":
            gen_kwargs.pop("return_full_text", None)

        if os.getenv("PATENT_EXTRACTOR_DEBUG_GEN", "").strip() == "1" and not getattr(
            owner, "_debug_gen_logged", False
        ):
            print(
                "[debug-gen] "
                f"model_class={owner.model.__class__.__name__} "
                f"pipeline_task={owner.pipeline_task} "
                f"gen_kwargs={gen_kwargs}"
            )
            owner._debug_gen_logged = True

        if owner.pipe is None or owner.pipeline_task == "direct-generate":
            gen_kwargs.pop("return_full_text", None)
            model_inputs = owner.tokenizer(prompt, return_tensors="pt")
            if hasattr(owner.model, "device"):
                try:
                    model_inputs = {
                        k: (v.to(owner.model.device) if hasattr(v, "to") else v)
                        for k, v in model_inputs.items()
                    }
                except Exception:
                    pass
            with torch.inference_mode():
                generated = owner.model.generate(**model_inputs, **gen_kwargs)
            input_ids = model_inputs.get("input_ids")
            if input_ids is not None and hasattr(input_ids, "shape") and generated.shape[-1] >= input_ids.shape[-1]:
                new_tokens = generated[0][input_ids.shape[-1] :]
                return owner.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return owner.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

        try:
            with torch.inference_mode():
                out = owner.pipe(prompt, **gen_kwargs)[0]
        except TypeError:
            gen_kwargs.pop("return_full_text", None)
            with torch.inference_mode():
                out = owner.pipe(prompt, **gen_kwargs)[0]
        return out.get("generated_text") or out.get("text") or ""

    if owner.backend == "vllm":
        sampling_params = build_vllm_sampling_params(
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

    raise ValueError(f"Unknown backend: {owner.backend}")
