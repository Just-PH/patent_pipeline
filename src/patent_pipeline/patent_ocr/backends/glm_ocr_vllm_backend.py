from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

_CLASSIFICATION_CUE_RE = re.compile(
    r"\b(?:klasse|classe|class(?:ification)?|int\.?\s*cl\.?)\b",
    flags=re.IGNORECASE,
)
_CLASSIFICATION_INLINE_RE = re.compile(
    r"\b(?:klasse|classe|class(?:ification)?|int\.?\s*cl\.?)\b[^\n]{0,48}?\d{1,3}\s*[a-z]?\b",
    flags=re.IGNORECASE,
)


def _preprocess_image(pil_img: Image.Image, mode: str) -> Image.Image:
    img = pil_img.convert("RGB")
    norm_mode = str(mode or "none").lower().strip()

    if norm_mode == "none":
        return img
    if norm_mode == "gray":
        return ImageOps.grayscale(img).convert("RGB")
    if norm_mode == "light":
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray)
        gray = ImageEnhance.Contrast(gray).enhance(1.05)
        return gray.convert("RGB")

    raise ValueError(f"Unknown preprocess mode: {mode!r}")


@dataclass
class GlmOcrVllmBackend:
    """
    GLM-OCR backend powered by offline vLLM batching.

    Intended usage:
      - segmentation_mode="backend"
      - workers=1
      - batch_size driven by cfg.ocr_config["batch_size"]
      - GPU-only workloads on the VM
    """

    model_name: str = "zai-org/GLM-OCR"
    prompt_text: str = "Text Recognition:"
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = True
    enforce_eager: bool = False
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    max_new_tokens: int = 4096
    temperature: float = 0.0
    resize_longest_edge: int = 1280
    name_: str = "glm-ocr-vllm"

    _llm: Any = field(init=False, default=None, repr=False)
    _sampling_params_cls: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if int(self.tensor_parallel_size) < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if not (0.0 < float(self.gpu_memory_utilization) <= 1.0):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_model_len is not None and int(self.max_model_len) < 1:
            raise ValueError("max_model_len must be >= 1")
        if self.max_num_seqs is not None and int(self.max_num_seqs) < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.max_num_batched_tokens is not None and int(self.max_num_batched_tokens) < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")
        if int(self.max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if int(self.resize_longest_edge) < 1:
            raise ValueError("resize_longest_edge must be >= 1")

    @property
    def name(self) -> str:
        return self.name_

    @property
    def is_gpu(self) -> bool:
        return True

    def _lazy_load(self) -> None:
        if self._llm is not None and self._sampling_params_cls is not None:
            return

        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            raise RuntimeError(
                "Missing deps for GLM-OCR vLLM backend. Build/run inside a vLLM image."
            ) from e

        llm_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "dtype": self.dtype,
            "tensor_parallel_size": int(self.tensor_parallel_size),
            "gpu_memory_utilization": float(self.gpu_memory_utilization),
            "enforce_eager": bool(self.enforce_eager),
            "limit_mm_per_prompt": {"image": 1},
        }
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = int(self.max_model_len)
        if self.tokenizer_mode != "auto":
            llm_kwargs["tokenizer_mode"] = self.tokenizer_mode
        if self.max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(self.max_num_seqs)
        if self.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = int(self.max_num_batched_tokens)

        self._llm = LLM(**llm_kwargs)
        self._sampling_params_cls = SamplingParams

    def _normalize_to_pil(self, img: Any) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        return Image.fromarray(np.asarray(img)).convert("RGB")

    def _resize_longest(self, pil: Image.Image, longest: int) -> Image.Image:
        if longest <= 0:
            return pil
        w, h = pil.size
        cur = max(w, h)
        if cur <= longest:
            return pil
        scale = longest / float(cur)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        return pil.resize((nw, nh), Image.BICUBIC)

    def _build_message(self, pil: Image.Image, prompt_text: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": pil},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def _build_sampling_params(
        self,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature if do_sample else 0.0),
        }
        return self._sampling_params_cls(**kwargs)

    def _resolve_prompt_texts(
        self,
        *,
        num_blocks: int,
        prompt_text: str,
        header_prompt_text: Optional[str],
        header_prompt_blocks: int,
    ) -> List[str]:
        prompts = [prompt_text] * num_blocks
        if not header_prompt_text or header_prompt_blocks <= 0:
            return prompts

        limit = min(num_blocks, header_prompt_blocks)
        for idx in range(limit):
            prompts[idx] = header_prompt_text
        return prompts

    def _run_prepared_ocr(
        self,
        *,
        prepared: List[Image.Image],
        prompt_texts: List[str],
        sampling_params: Any,
        batch_size: int,
    ) -> List[str]:
        outs: List[str] = []
        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i : i + batch_size]
            chunk_prompts = prompt_texts[i : i + batch_size]
            messages = [
                self._build_message(pil, chunk_prompt)
                for pil, chunk_prompt in zip(chunk, chunk_prompts)
            ]
            outputs = self._llm.chat(messages=messages, sampling_params=sampling_params)

            if len(outputs) != len(chunk):
                raise RuntimeError(
                    f"GLM-OCR vLLM backend returned {len(outputs)} texts for {len(chunk)} images"
                )

            for output in outputs:
                candidates = getattr(output, "outputs", None) or []
                text = ""
                if candidates:
                    text = str(getattr(candidates[0], "text", "") or "")
                outs.append(text.strip())

        return outs

    def _has_classification_text(self, text: str) -> bool:
        if not text:
            return False
        return bool(_CLASSIFICATION_INLINE_RE.search(text))

    def _extract_classification_snippets(self, text: str) -> List[str]:
        snippets: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _CLASSIFICATION_CUE_RE.search(line) and any(ch.isdigit() for ch in line):
                snippets.append(line)

        if not snippets:
            snippets.extend(match.group(0).strip() for match in _CLASSIFICATION_INLINE_RE.finditer(text))

        out: List[str] = []
        seen = set()
        for snippet in snippets:
            key = snippet.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        if not block_imgs:
            return []

        self._lazy_load()

        preprocess_mode = str(ocr_config.get("preprocess", "none"))
        prompt_text = str(ocr_config.get("prompt_text", self.prompt_text))
        header_prompt_raw = ocr_config.get("header_prompt_text")
        header_prompt_text = None if header_prompt_raw in (None, "") else str(header_prompt_raw)
        header_prompt_blocks = int(
            ocr_config.get("header_prompt_blocks", 1 if header_prompt_text else 0)
        )
        max_new_tokens = int(ocr_config.get("max_new_tokens", self.max_new_tokens))
        classification_fallback_prompt_raw = ocr_config.get("classification_fallback_prompt_text")
        classification_fallback_prompt_text = (
            None
            if classification_fallback_prompt_raw in (None, "")
            else str(classification_fallback_prompt_raw)
        )
        classification_fallback_blocks = int(
            ocr_config.get(
                "classification_fallback_blocks",
                1 if classification_fallback_prompt_text else 0,
            )
        )
        classification_fallback_max_new_tokens = int(
            ocr_config.get(
                "classification_fallback_max_new_tokens",
                min(max_new_tokens, 256) if classification_fallback_prompt_text else max_new_tokens,
            )
        )
        do_sample = bool(ocr_config.get("do_sample", False))
        temperature = float(ocr_config.get("temperature", self.temperature))
        batch_size = int(ocr_config.get("batch_size", len(block_imgs)))
        resize_longest_edge = int(ocr_config.get("resize_longest_edge", self.resize_longest_edge))

        sampling_params = self._build_sampling_params(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        prepared: List[Image.Image] = []
        for img in block_imgs:
            pil = self._normalize_to_pil(img)
            pil = _preprocess_image(pil, preprocess_mode)
            pil = self._resize_longest(pil, resize_longest_edge)
            prepared.append(pil)

        prompt_texts = self._resolve_prompt_texts(
            num_blocks=len(prepared),
            prompt_text=prompt_text,
            header_prompt_text=header_prompt_text,
            header_prompt_blocks=header_prompt_blocks,
        )

        outs = self._run_prepared_ocr(
            prepared=prepared,
            prompt_texts=prompt_texts,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )

        combined_text = "\n\n".join(t for t in outs if t).strip()
        if (
            classification_fallback_prompt_text
            and classification_fallback_blocks > 0
            and not self._has_classification_text(combined_text)
        ):
            fallback_count = min(len(prepared), classification_fallback_blocks)
            fallback_prepared = prepared[:fallback_count]
            fallback_prompt_texts = [classification_fallback_prompt_text] * fallback_count
            fallback_sampling_params = self._build_sampling_params(
                max_new_tokens=classification_fallback_max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
            fallback_outs = self._run_prepared_ocr(
                prepared=fallback_prepared,
                prompt_texts=fallback_prompt_texts,
                sampling_params=fallback_sampling_params,
                batch_size=min(batch_size, fallback_count),
            )
            fallback_text = "\n\n".join(t for t in fallback_outs if t).strip()
            snippets = [
                snippet
                for snippet in self._extract_classification_snippets(fallback_text)
                if snippet not in combined_text
            ]
            if snippets:
                prefix = "\n".join(snippets).strip()
                first_text = outs[0].strip()
                outs[0] = f"{prefix}\n\n{first_text}".strip() if first_text else prefix

        if len(outs) != len(block_imgs):
            raise RuntimeError(
                f"GLM-OCR vLLM backend returned {len(outs)} texts for {len(block_imgs)} images"
            )
        return outs

    def validate_ocr_config(self, ocr_config: Dict[str, Any]) -> None:
        if not isinstance(ocr_config, dict):
            raise TypeError("ocr_config must be a dict")

        mode = str(ocr_config.get("preprocess", "none")).lower().strip()
        if mode not in {"none", "gray", "light"}:
            raise ValueError("ocr_config['preprocess'] must be one of: none|gray|light")

        if "prompt_text" in ocr_config and not isinstance(ocr_config["prompt_text"], str):
            raise TypeError("ocr_config['prompt_text'] must be str")
        if "header_prompt_text" in ocr_config and not isinstance(
            ocr_config["header_prompt_text"], str
        ):
            raise TypeError("ocr_config['header_prompt_text'] must be str")
        if "classification_fallback_prompt_text" in ocr_config and not isinstance(
            ocr_config["classification_fallback_prompt_text"], str
        ):
            raise TypeError("ocr_config['classification_fallback_prompt_text'] must be str")

        for key in (
            "max_new_tokens",
            "batch_size",
            "resize_longest_edge",
            "header_prompt_blocks",
            "classification_fallback_blocks",
            "classification_fallback_max_new_tokens",
        ):
            if key in ocr_config:
                value = ocr_config[key]
                if not isinstance(value, int):
                    raise ValueError(f"ocr_config['{key}'] must be an int")
                if key in {"header_prompt_blocks", "classification_fallback_blocks"}:
                    if int(value) < 0:
                        raise ValueError(f"ocr_config['{key}'] must be >= 0")
                elif int(value) <= 0:
                    raise ValueError(f"ocr_config['{key}'] must be a positive int")

        if "temperature" in ocr_config and not isinstance(ocr_config["temperature"], (int, float)):
            raise TypeError("ocr_config['temperature'] must be float")
        if "do_sample" in ocr_config and not isinstance(ocr_config["do_sample"], bool):
            raise TypeError("ocr_config['do_sample'] must be bool")
