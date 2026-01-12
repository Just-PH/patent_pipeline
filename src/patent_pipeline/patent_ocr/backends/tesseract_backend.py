from typing import Any, Dict, List
from PIL import Image
import numpy as np
import cv2
import pytesseract


class TesseractBackend:
    """
    OCR backend for Tesseract.
    Pure backend: no OcrEngine, no indirection.
    """

    def __init__(self, *, is_gpu: bool = False):
        # tesseract is always CPU, but keep signature uniform
        self._is_gpu = False
        self._name = "tesseract"

    @property
    def is_gpu(self) -> bool:
        return self._is_gpu

    @property
    def name(self) -> str:
        return self._name

    # --- preprocess (local to backend)
    def _preprocess(self, pil_img: Image.Image, cfg: Dict[str, Any]) -> Image.Image:
        mode = cfg.get("preprocess", "gray")

        if mode == "none":
            return pil_img

        gray = np.array(pil_img.convert("L"))

        if mode == "gray":
            return Image.fromarray(gray)

        if mode == "light":
            gray = cv2.GaussianBlur(gray, (1, 1), 0)
            return Image.fromarray(gray)

        raise ValueError(f"Unknown preprocess mode for Tesseract: {mode}")

    # --- OCR execution (PIPELINE ENTRYPOINT)
    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        results: List[str] = []

        lang = ocr_config["lang"]
        cfg = ocr_config["config"]

        for img in block_imgs:
            pil_img = self._preprocess(img, ocr_config)
            txt = pytesseract.image_to_string(
                pil_img,
                lang=lang,
                config=cfg,
            )
            results.append((txt or "").strip())

        return results
