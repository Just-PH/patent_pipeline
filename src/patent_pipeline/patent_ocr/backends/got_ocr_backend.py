# src/patent_pipeline/patent_ocr/backends/got_ocr_backend.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from patent_pipeline.patent_ocr.utils.image_preprocess import preprocess_pil


@dataclass
class GotOcrBackend:
    """
    GOT-OCR2 backend (Transformers) - aligned with DocTR/Tesseract backend style.

    Exposes:
      - .name
      - .is_gpu
      - run_blocks_ocr(block_imgs, ocr_config) -> List[str]
    """

    model_name: str = "stepfun-ai/GOT-OCR-2.0-hf"
    device: str = "cpu"           # "cpu" | "cuda" | "mps" | "auto"
    dtype: Optional[str] = None   # "float16" | "bfloat16" | "float32" | None
    trust_remote_code: bool = True
    name_: str = "got-ocr"

    _processor: Any = field(init=False, default=None, repr=False)
    _model: Any = field(init=False, default=None, repr=False)
    _torch: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.device = (self.device or "cpu").lower().strip()

    @property
    def name(self) -> str:
        return self.name_

    @property
    def is_gpu(self) -> bool:
        return self.device in {"cuda", "mps"}

    # -------------------------
    # internals
    # -------------------------
    def _resolve_device(self, torch) -> str:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"Invalid device='{self.device}'. Use 'cpu'|'cuda'|'mps'|'auto'.")
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, device: str, torch):
        dt = self.dtype
        if dt is None:
            if device == "cuda":
                dt = "float16"
            elif device == "mps":
                # si MPS fait n'importe quoi -> passe dtype="float32"
                dt = "float16"
            else:
                dt = "float32"
        mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if dt not in mapping:
            raise ValueError(f"Invalid dtype='{dt}'. Use 'float16'|'bfloat16'|'float32' or None.")
        return mapping[dt]

    def _lazy_load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except Exception as e:
            raise RuntimeError(
                "Missing deps for GOT-OCR2 backend. Try:\n"
                "  pip install -U torch transformers accelerate sentencepiece\n"
            ) from e

        self._torch = torch

        resolved_device = self._resolve_device(torch)
        self.device = resolved_device
        torch_dtype = self._resolve_dtype(resolved_device, torch)

        # GOT-OCR2 uses AutoModelForImageTextToText per transformers docs
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self._model.eval().to(self.device)

        # use_fast=True est recommandé dans la doc GOT-OCR2
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )

    # -------------------------
    # public API (same idea as doctr/tesseract)
    # -------------------------
    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        """
        Args:
          block_imgs: list of PIL.Image or array-like
          ocr_config:
            - preprocess: "none" | "gray" | "light"
            - max_new_tokens: int (default 1024)
            - stop_strings: str (default "<|im_end|>")
            - format: bool (default False)  # formatted OCR mode (markdown/latex). Plain by default.
        """
        if not block_imgs:
            return []

        self._lazy_load()
        torch = self._torch

        preprocess_mode = ocr_config.get("preprocess", "none")
        max_new_tokens = int(ocr_config.get("max_new_tokens", 1024))
        stop_strings = ocr_config.get("stop_strings", "<|im_end|>")
        format_flag = bool(ocr_config.get("format", False))

        out: List[str] = []

        for im in block_imgs:
            # normalize to PIL RGB
            if isinstance(im, Image.Image):
                pil = im.convert("RGB")
            else:
                arr = np.asarray(im)
                if arr.ndim == 2:
                    pil = Image.fromarray(arr).convert("RGB")
                else:
                    pil = Image.fromarray(arr).convert("RGB")

            # shared preprocess hook
            pil = preprocess_pil(pil, mode=preprocess_mode)

            # processor -> tensors
            inputs = self._processor(
                images=pil,
                return_tensors="pt",
                format=format_flag,
            )
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.inference_mode():
                gen_ids = self._model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=self._processor.tokenizer,  # important for GOT
                    stop_strings=stop_strings,            # important for GOT
                    max_new_tokens=max_new_tokens,
                )

            # GOT decode: remove prompt tokens (input_ids prefix)
            # (pattern from official doc)
            prompt_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
            decoded = self._processor.decode(
                gen_ids[0, prompt_len:],
                skip_special_tokens=True,
            )
            out.append((decoded or "").strip())

        if len(out) != len(block_imgs):
            raise RuntimeError(f"GOT-OCR2 returned {len(out)} texts for {len(block_imgs)} images")

        return out

    def validate_ocr_config(self, ocr_config: Dict[str, Any]) -> None:
        # model_name est crucial
        if not getattr(self, "model_name", None):
            raise ValueError(
                "[GotOcrBackend] 'model_name' is required (backend init). "
                "Example: GotOcrBackend(model_name='ucaslcl/GOT-OCR2_0')"
            )

        dev = str(getattr(self, "device", "cpu")).lower().strip()
        if dev not in {"cpu", "mps", "cuda"}:
            raise ValueError("[GotOcrBackend] device must be one of: cpu|mps|cuda")

        # Typical generation sanity
        if "max_new_tokens" in ocr_config:
            mnt = ocr_config["max_new_tokens"]
            if not isinstance(mnt, int) or mnt <= 0:
                raise ValueError("[GotOcrBackend] ocr_config['max_new_tokens'] must be a positive int.")

        mode = str(ocr_config.get("preprocess", "none")).lower().strip()
        if mode not in {"none", "gray", "light"}:
            raise ValueError("[GotOcrBackend] Invalid ocr_config['preprocess'] (use none|gray|light).")
