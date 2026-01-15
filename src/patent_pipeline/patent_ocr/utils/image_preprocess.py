# src/patent_pipeline/patent_ocr/utils/image_preprocess.py

from typing import Literal
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2

PreprocessMode = Literal["none", "gray", "light"]


def preprocess_pil(
    pil_img: Image.Image,
    mode: PreprocessMode = "none",
) -> Image.Image:
    """
    Shared image preprocessing for OCR backends.

    This function is intentionally conservative.
    It should NEVER destroy information.

    Parameters
    ----------
    pil_img : PIL.Image
        Input image (RGB or grayscale).
    mode : {"none", "gray", "light"}
        Preprocessing mode.

    Returns
    -------
    PIL.Image
        Preprocessed RGB image.
    """
    img = pil_img.convert("RGB")

    if mode == "none":
        return img

    if mode == "gray":
        return ImageOps.grayscale(img).convert("RGB")

    if mode == "light":
        # Gentle preprocessing safe for OCR:
        # - grayscale
        # - light denoising
        # - slight contrast normalization
        gray = np.array(ImageOps.grayscale(img))

        gray = cv2.GaussianBlur(gray, (1, 1), 0)
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        out = Image.fromarray(gray)
        return out.convert("RGB")

    raise ValueError(f"Unknown preprocess mode: {mode}")
