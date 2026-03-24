# src/patent_pipeline/patent_ocr/backends/surya_backend.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from patent_pipeline.patent_ocr.utils.image_preprocess import preprocess_pil


def _to_pil_rgb(im: Any) -> Image.Image:
    """Accept PIL.Image or numpy-like arrays; return PIL RGB."""
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    return Image.fromarray(im).convert("RGB")


def _extract_lines_text(pred_obj: Any) -> List[str]:
    """
    Surya Python API may return:
      - objects with .text_lines, each line having .text
      - dicts with ["text_lines"], each line dict having ["text"]
    Normalize to list[str] lines (already in reading order as provided by Surya).
    """
    out: List[str] = []

    # object-style
    if hasattr(pred_obj, "text_lines"):
        tl = getattr(pred_obj, "text_lines") or []
        for ln in tl:
            txt = ""
            if ln is not None:
                txt = getattr(ln, "text", "") or ""
            txt = txt.strip()
            if txt:
                out.append(txt)
        return out

    # dict-style
    if isinstance(pred_obj, dict):
        for ln in (pred_obj.get("text_lines") or []):
            if isinstance(ln, dict):
                txt = (ln.get("text") or "").strip()
                if txt:
                    out.append(txt)
        return out

    return out


def _get_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _set_env_int(key: str, value: Optional[int]) -> None:
    """Set env var to int string if value is not None."""
    if value is None:
        return
    os.environ[key] = str(int(value))


@dataclass
class SuryaOcrBackend:
    """
    Surya OCR backend (GPU/CPU) for patent_pipeline.

    Expected interface (like your other backends):
      - name: str
      - is_gpu: bool
      - validate_ocr_config(ocr_config)
      - run_blocks_ocr(block_imgs, ocr_config) -> list[str]

    Notes:
      - Surya uses env vars RECOGNITION_BATCH_SIZE / DETECTOR_BATCH_SIZE for perf tuning.
      - We support passing those via ocr_config:
          recognition_batch_size: int
          detector_batch_size: int
      - We apply modest defaults if not provided.
    """

    device: str = "auto"  # "auto" | "cuda" | "cpu" | "mps" (best-effort)
    name_: str = "surya"

    # lazy loaded predictors
    _foundation: Any = None
    _recognizer: Any = None
    _detector: Any = None
    _resolved_device: str = "cpu"

    def __post_init__(self) -> None:
        # Import here so project can import without surya installed
        import torch
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        req = (self.device or "auto").strip().lower()

        cuda_ok = torch.cuda.is_available()
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if req in {"auto", "gpu"}:
            if cuda_ok:
                self._resolved_device = "cuda"
            elif mps_ok:
                self._resolved_device = "mps"
            else:
                self._resolved_device = "cpu"
        else:
            # explicit request
            self._resolved_device = req

        # safety fallback
        if self._resolved_device == "cuda" and not cuda_ok:
            self._resolved_device = "cpu"
        if self._resolved_device == "mps" and not mps_ok:
            self._resolved_device = "cpu"

        # Build predictors
        self._foundation = FoundationPredictor()
        self._recognizer = RecognitionPredictor(self._foundation)
        self._detector = DetectionPredictor()

    @property
    def name(self) -> str:
        return self.name_

    @property
    def is_gpu(self) -> bool:
        return self._resolved_device in {"cuda", "mps"}

    def validate_ocr_config(self, ocr_config: Dict[str, Any]) -> None:
        # preprocess
        mode = str(ocr_config.get("preprocess", "none")).strip().lower()
        allowed = {"none", "gray", "light"}
        if mode not in allowed:
            raise ValueError(
                f"[SuryaOcrBackend] Invalid ocr_config['preprocess']={mode!r}. Allowed: {sorted(allowed)}"
            )

        # langs
        langs = ocr_config.get("langs", None)
        if langs is not None and not isinstance(langs, (list, tuple, str)):
            raise TypeError("[SuryaOcrBackend] ocr_config['langs'] must be list[str] or 'de,en' string.")
        if isinstance(langs, (list, tuple)) and not all(isinstance(x, str) for x in langs):
            raise TypeError("[SuryaOcrBackend] ocr_config['langs'] list must contain only strings.")

        # batch knobs (flat)
        for k in ("recognition_batch_size", "detector_batch_size"):
            v = ocr_config.get(k, None)
            if v is None:
                continue
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"[SuryaOcrBackend] {k} must be a positive int (got {v!r}).")

        # optional: let users explicitly disable env override (rare)
        disable_env = ocr_config.get("disable_env_batch_override", None)
        if disable_env is not None and not isinstance(disable_env, (bool, int, float, str)):
            raise TypeError("[SuryaOcrBackend] disable_env_batch_override must be bool-like if provided.")

    def _resolve_langs(self, ocr_config: Dict[str, Any]) -> Optional[List[str]]:
        langs = ocr_config.get("langs", None)
        if langs is None:
            return None
        if isinstance(langs, str):
            langs = [x.strip() for x in langs.split(",") if x.strip()]
            return langs or None
        # list/tuple
        langs = [x.strip() for x in langs if isinstance(x, str) and x.strip()]
        return langs or None

    def _apply_batch_env(self, ocr_config: Dict[str, Any]) -> None:
        """
        Surya reads these env vars for batching/perf.
        We set modest defaults if not provided, unless disable_env_batch_override is true.
        """
        if _get_bool(ocr_config.get("disable_env_batch_override", False)):
            return

        if self.is_gpu:
            default_rec = 16
            default_det = 8
        else:
            default_rec = 4
            default_det = 2

        rec_bs = ocr_config.get("recognition_batch_size", None)
        det_bs = ocr_config.get("detector_batch_size", None)

        if rec_bs is None:
            rec_bs = default_rec
        if det_bs is None:
            det_bs = default_det

        _set_env_int("RECOGNITION_BATCH_SIZE", rec_bs)
        _set_env_int("DETECTOR_BATCH_SIZE", det_bs)

    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        if not block_imgs:
            return []

        # validate config early
        self.validate_ocr_config(ocr_config)

        # apply env knobs (batch sizes) before calling predictors
        self._apply_batch_env(ocr_config)

        preprocess_mode = str(ocr_config.get("preprocess", "none")).strip().lower()
        langs = self._resolve_langs(ocr_config)

        # prepare images
        images: List[Image.Image] = []
        for im in block_imgs:
            pil = _to_pil_rgb(im)
            pil = preprocess_pil(pil, mode=preprocess_mode)
            images.append(pil)

        # predict
        try:
            if langs is None:
                preds = self._recognizer(images, det_predictor=self._detector)
            else:
                # Some versions support langs=...; best-effort
                preds = self._recognizer(images, det_predictor=self._detector, langs=langs)
        except TypeError:
            # fallback if langs kw not supported
            preds = self._recognizer(images, det_predictor=self._detector)

        # normalize to texts
        texts: List[str] = []
        for pred in (preds or []):
            lines = _extract_lines_text(pred)
            texts.append("\n".join(lines).strip())

        if len(texts) != len(block_imgs):
            raise RuntimeError(
                f"[SuryaOcrBackend] Returned {len(texts)} texts for {len(block_imgs)} images."
            )
        return texts
