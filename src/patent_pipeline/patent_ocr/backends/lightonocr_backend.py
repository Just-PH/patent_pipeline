# src/patent_pipeline/patent_ocr/backends/lightonocr_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import warnings

from PIL import Image

from patent_pipeline.patent_ocr.backends.deps import LIGHTONOCR_DEPS, import_module_with_auto_install


@dataclass
class LightOnOcrBackend:
    """
    LightOnOCR-2-1B backend (Transformers).

    Notes:
      - Designed to run in Pipeline_OCR with segmentation_mode="backend"
        (whole page passed as a single block).
      - Requires a transformers version that includes LightOnOCR support
        (model card currently recommends installing transformers from source).
    """

    model_id: str = "lightonai/LightOnOCR-2-1B"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    dtype: str = "auto"   # "auto" | "float32" | "bfloat16" | "float16"
    max_new_tokens: int = 4096
    # Main throughput lever for GPU mode (pipeline already uses workers=1 for GPU backends).
    batch_size: int = 1
    auto_install_deps: bool = False

    # Optional perf knobs
    compile: bool = False  # torch.compile on supported setups (best-effort)

    # Lazy-loaded
    _model: Any = None
    _processor: Any = None
    _torch: Any = None
    _resolved_device: Optional[str] = None
    _resolved_dtype: Optional[Any] = None

    def __post_init__(self) -> None:
        # Do nothing heavy here: keep lazy-load.
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be > 0")

    @property
    def name(self) -> str:
        return "lightonocr"

    @property
    def is_gpu(self) -> bool:
        d = (self._resolved_device or self.device or "").lower()
        return d in {"cuda", "mps"}

    # -------------------------
    # Internal helpers
    # -------------------------
    def _resolve_device(self, torch) -> str:
        if self.device and self.device != "auto":
            return self.device

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_dtype(self, device: str, torch):
        if self.dtype and self.dtype != "auto":
            m = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            if self.dtype not in m:
                raise ValueError(f"Unsupported dtype={self.dtype!r}. Choose among {sorted(m)} or 'auto'.")
            return m[self.dtype]

        # Model card recommendation: float32 on mps, bfloat16 elsewhere
        if device == "mps":
            return torch.float32
        return torch.bfloat16

    def _resize_longest_edge(self, pil: Image.Image, longest: int) -> Image.Image:
        if longest <= 0:
            return pil
        w, h = pil.size
        cur = max(w, h)
        if cur <= longest:
            return pil
        scale = longest / float(cur)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        return pil.resize((nw, nh), Image.BICUBIC)

    def _lazy_load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        torch = import_module_with_auto_install(
            module_name="torch",
            backend_name="LightOnOCR",
            deps=LIGHTONOCR_DEPS,
            auto_install=self.auto_install_deps,
            err_hint="LightOnOCR backend requires torch.",
        )
        transformers_module = import_module_with_auto_install(
            module_name="transformers",
            backend_name="LightOnOCR",
            deps=LIGHTONOCR_DEPS,
            auto_install=self.auto_install_deps,
            err_hint="LightOnOCR backend requires a transformers release that includes LightOnOCR.",
        )
        LightOnOcrForConditionalGeneration = getattr(
            transformers_module, "LightOnOcrForConditionalGeneration"
        )
        LightOnOcrProcessor = getattr(transformers_module, "LightOnOcrProcessor")

        self._torch = torch
        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(device, torch)

        model = LightOnOcrForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=dtype)
        model = model.to(device)

        # Best-effort compile (only if explicitly requested)
        if self.compile:
            try:
                model = torch.compile(model)  # type: ignore[attr-defined]
            except Exception:
                # compile is optional; ignore if not supported
                pass

        processor = LightOnOcrProcessor.from_pretrained(self.model_id)

        self._model = model
        self._processor = processor
        self._resolved_device = device
        self._resolved_dtype = dtype

    # -------------------------
    # Public API expected by Pipeline_OCR
    # -------------------------
    def validate_ocr_config(self, ocr_config: Dict[str, Any]) -> None:
        """
        Supported keys (all optional):
          - max_new_tokens: int (overrides constructor)
          - resize_longest_edge: int (default 1540 if absent)
          - temperature: float (default 0.0 -> deterministic)
          - do_sample: bool (default False)
          - batch_size: int > 0 (overrides constructor)
        """
        if not isinstance(ocr_config, dict):
            raise TypeError("ocr_config must be a dict")

        if "max_new_tokens" in ocr_config and not isinstance(ocr_config["max_new_tokens"], int):
            raise TypeError("ocr_config['max_new_tokens'] must be int")
        if "resize_longest_edge" in ocr_config and not isinstance(ocr_config["resize_longest_edge"], int):
            raise TypeError("ocr_config['resize_longest_edge'] must be int")
        if "temperature" in ocr_config and not isinstance(ocr_config["temperature"], (int, float)):
            raise TypeError("ocr_config['temperature'] must be float")
        if "do_sample" in ocr_config and not isinstance(ocr_config["do_sample"], bool):
            raise TypeError("ocr_config['do_sample'] must be bool")
        if "batch_size" in ocr_config:
            if not isinstance(ocr_config["batch_size"], int):
                raise TypeError("ocr_config['batch_size'] must be int")
            if int(ocr_config["batch_size"]) <= 0:
                raise ValueError("ocr_config['batch_size'] must be > 0")

    def _prepare_image(self, img: Any, resize_longest_edge: int) -> Image.Image:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        pil = img.convert("RGB")
        return self._resize_longest_edge(pil, resize_longest_edge)

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = self._resolved_device
        dtype = self._resolved_dtype
        moved: Dict[str, Any] = {}
        for k, v in inputs.items():
            if getattr(v, "is_floating_point", lambda: False)():
                moved[k] = v.to(device=device, dtype=dtype)
            else:
                moved[k] = v.to(device)
        return moved

    def _decode_batch(self, output_ids, inputs: Dict[str, Any], processor) -> List[str]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch = int(output_ids.shape[0])
        texts: List[str] = []
        for i in range(batch):
            if attention_mask is not None:
                prompt_len = int(attention_mask[i].sum().item())
            else:
                prompt_len = int(input_ids[i].shape[0])
            generated_ids = output_ids[i, prompt_len:]
            texts.append(processor.decode(generated_ids, skip_special_tokens=True))
        return texts

    def _generate_batch(
        self,
        pil_imgs: List[Image.Image],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> List[str]:
        torch = self._torch
        model = self._model
        processor = self._processor

        conversations = [
            [{"role": "user", "content": [{"type": "image", "image": pil}]}]
            for pil in pil_imgs
        ]

        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs_to_device(inputs)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(1e-6, temperature)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        return self._decode_batch(output_ids, inputs, processor)

    def _generate_single(
        self,
        pil: Image.Image,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> str:
        texts = self._generate_batch(
            [pil],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        return texts[0] if texts else ""

    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        """
        Expects PIL images in block_imgs (Pipeline_OCR already passes PIL).
        Returns list[str] aligned with block_imgs.

        In segmentation_mode="backend", block_imgs will be [page_img].
        """
        self._lazy_load()

        max_new_tokens = int(ocr_config.get("max_new_tokens", self.max_new_tokens))
        resize_longest_edge = int(ocr_config.get("resize_longest_edge", 1540))
        do_sample = bool(ocr_config.get("do_sample", False))
        temperature = float(ocr_config.get("temperature", 0.0))
        batch_size = int(ocr_config.get("batch_size", self.batch_size))

        prepared = [self._prepare_image(img, resize_longest_edge) for img in block_imgs]
        outs: List[str] = []

        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i : i + batch_size]

            try:
                batch_out = self._generate_batch(
                    chunk,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )
            except Exception as e:
                warnings.warn(
                    f"[LightOnOCR] Batch path failed on chunk starting at index {i}; "
                    f"falling back to item-by-item. Cause: {type(e).__name__}: {e}"
                )
                batch_out = []
                for pil in chunk:
                    try:
                        batch_out.append(
                            self._generate_single(
                                pil,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                temperature=temperature,
                            )
                        )
                    except Exception as one_err:
                        warnings.warn(
                            "[LightOnOCR] Item fallback failed; returning empty text for one sample. "
                            f"Cause: {type(one_err).__name__}: {one_err}"
                        )
                        batch_out.append("")

            if len(batch_out) != len(chunk):
                warnings.warn(
                    f"[LightOnOCR] Unexpected batch output size {len(batch_out)} for chunk {len(chunk)}; "
                    "repairing with empty strings."
                )
                if len(batch_out) < len(chunk):
                    batch_out = batch_out + ([""] * (len(chunk) - len(batch_out)))
                else:
                    batch_out = batch_out[: len(chunk)]

            outs.extend(batch_out)

        return outs
