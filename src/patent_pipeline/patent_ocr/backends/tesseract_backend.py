# src/patent_pipeline/patent_ocr/backends/tesseract_backend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from PIL import Image
import pytesseract

from patent_pipeline.patent_ocr.utils.image_preprocess import preprocess_pil


@dataclass
class TesseractBackend:
    """
    Tesseract OCR backend (CPU).

    - Uses shared preprocess_pil(...) to avoid duplication across backends.
    - Works best with segmentation_mode="custom" on difficult layouts / fraktur.
    """

    name_: str = "tesseract"

    @property
    def is_gpu(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self.name_

    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        if not block_imgs:
            return []

        lang = ocr_config.get("lang", "")
        config = ocr_config.get("config", "")
        preprocess_mode = ocr_config.get("preprocess", "none")

        out: List[str] = []
        for im in block_imgs:
            pil = im if isinstance(im, Image.Image) else Image.fromarray(im).convert("RGB")

            # ✅ shared preprocess
            pil = preprocess_pil(pil, mode=preprocess_mode)

            txt = pytesseract.image_to_string(pil, lang=lang, config=config)
            out.append((txt or "").strip())

        if len(out) != len(block_imgs):
            raise RuntimeError(f"Tesseract backend returned {len(out)} texts for {len(block_imgs)} images")

        return out

    def validate_ocr_config(self, ocr_config: Dict[str, Any]) -> None:
        lang = ocr_config.get("lang")
        config = ocr_config.get("config")

        missing = []
        if not lang:
            missing.append("lang")
        if not config:
            missing.append("config")

        if missing:
            raise ValueError(
                "[TesseractBackend] Missing required ocr_config keys: "
                f"{missing}. Example:\n"
                "  ocr_config={"
                "\"lang\":\"frk+deu\","
                "\"config\":\"--psm 6 --oem 1 -c preserve_interword_spaces=1\","
                "\"preprocess\":\"light\"}"
            )

        if not isinstance(lang, str):
            raise TypeError("[TesseractBackend] ocr_config['lang'] must be a string.")
        if not isinstance(config, str):
            raise TypeError("[TesseractBackend] ocr_config['config'] must be a string.")
