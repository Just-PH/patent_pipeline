# src/patent_pipeline/patent_ocr/backends/trocr_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from patent_pipeline.patent_ocr.utils.image_preprocess import preprocess_pil


def _normalize_device(device: str) -> str:
    d = (device or "cpu").lower().strip()
    if d not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"device must be one of: cpu|cuda|mps (got {device!r})")
    if d == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    if d == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but torch.backends.mps.is_available() is False")
    return d


def _pick_dtype(device: str, prefer_fp16: bool) -> torch.dtype:
    if not prefer_fp16:
        return torch.float32
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


@dataclass
class TrOCRBackend:
    """
    TrOCR backend (PyTorch).

    - TrOCR = reconnaissance (image -> texte), PAS de détection/layout.
    - Donc: utilise-le surtout avec segmentation_mode="custom" (tes crops).
    - Le preprocess est centralisé dans utils.image_preprocess.preprocess_pil
      pour éviter toute redondance entre backends.
    """

    device: str = "cpu"
    model_name: str = "microsoft/trocr-large-printed"
    name_: str = "trocr"

    # perf / decoding
    batch_size: int = 8
    num_beams: int = 1
    max_new_tokens: int = 256
    prefer_fp16: bool = True

    # limite de taille pour éviter des crops énormes
    max_side: Optional[int] = 1600

    def __post_init__(self) -> None:
        self.device = _normalize_device(self.device)
        self.dtype = _pick_dtype(self.device, self.prefer_fp16)

        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

        if self.device == "cuda":
            self.model = self.model.to("cuda")
        elif self.device == "mps":
            self.model = self.model.to("mps")

        if self.dtype != torch.float32:
            self.model = self.model.to(dtype=self.dtype)

        self.model.eval()

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

    @property
    def is_gpu(self) -> bool:
        return self.device in {"cuda", "mps"}

    @property
    def name(self) -> str:
        return self.name_

    def _maybe_resize(self, img: Image.Image) -> Image.Image:
        if not self.max_side:
            return img
        w, h = img.size
        m = max(w, h)
        if m <= self.max_side:
            return img
        scale = self.max_side / float(m)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), resample=Image.BICUBIC)

    @torch.no_grad()
    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        if not block_imgs:
            return []

        # overrides runtime
        bs = int(ocr_config.get("batch_size", self.batch_size))
        num_beams = int(ocr_config.get("num_beams", self.num_beams))
        max_new_tokens = int(ocr_config.get("max_new_tokens", self.max_new_tokens))

        # preprocess shared (centralisé)
        preprocess_mode = ocr_config.get("preprocess", "none")

        # convert + preprocess
        imgs: List[Image.Image] = []
        for im in block_imgs:
            pil = im if isinstance(im, Image.Image) else Image.fromarray(im).convert("RGB")

            # ✅ ICI: preprocess commun (plus de _preprocess_pil local)
            pil = preprocess_pil(pil, mode=preprocess_mode)

            pil = self._maybe_resize(pil)
            imgs.append(pil)

        texts: List[str] = []

        for i in range(0, len(imgs), bs):
            batch_imgs = imgs[i : i + bs]

            enc = self.processor(images=batch_imgs, return_tensors="pt")
            pixel_values = enc.pixel_values

            if self.device == "cuda":
                pixel_values = pixel_values.to("cuda", dtype=self.dtype)
            elif self.device == "mps":
                pixel_values = pixel_values.to("mps", dtype=self.dtype)
            else:
                pixel_values = pixel_values.to("cpu")

            gen_ids = self.model.generate(
                pixel_values,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )

            batch_texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
            texts.extend([(t or "").strip() for t in batch_texts])

        if len(texts) != len(block_imgs):
            raise RuntimeError(f"TrOCR backend returned {len(texts)} texts for {len(block_imgs)} images")

        return texts
