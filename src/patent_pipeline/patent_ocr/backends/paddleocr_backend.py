# src/patent_pipeline/patent_ocr/backends/paddleocr_backend.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from patent_pipeline.patent_ocr.utils.image_preprocess import preprocess_pil


@dataclass
class PaddleOcrBackend:
    """
    PaddleOCR backend aligned with your DocTR/Tesseract backend interface.

    PaddleOCR has multiple API variants depending on version / packaging:
      - Classic API: engine.ocr(img, det=..., rec=..., cls=...)
      - Newer / pipeline API: engine.ocr(...) is deprecated and proxies to engine.predict(...)
        and predict() may NOT accept det/rec/cls kwargs at all.

    This backend is robust to both by:
      - Trying engine.ocr(..., det=..., rec=..., cls=...)
      - If it fails with "unexpected keyword argument", falling back to engine.predict(img)
        (no det/rec/cls kwargs).

    Exposes:
      - .name
      - .is_gpu
      - run_blocks_ocr(block_imgs, ocr_config) -> List[str]
    """

    # Engine configuration
    lang: str = "en"
    use_angle_cls: bool = False
    use_gpu: bool = False  # may be ignored depending on PaddleOCR version/platform
    name_: str = "paddleocr"

    _engine: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.lang = (self.lang or "en").lower().strip()

    @property
    def name(self) -> str:
        return self.name_

    @property
    def is_gpu(self) -> bool:
        # Pipeline uses this to avoid naive multiprocessing for GPU backends.
        return bool(self.use_gpu)

    # ---------------------------------------------------------------------
    # Lazy init
    # ---------------------------------------------------------------------
    def _lazy_load(self) -> None:
        if self._engine is not None:
            return

        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            raise RuntimeError(
                "Missing deps for PaddleOCR backend.\n"
                "Install in the SAME interpreter as your notebook:\n"
                "  python -m pip install -U paddlepaddle paddleocr\n"
            ) from e

        # PaddleOCR constructor kwargs vary across versions/platforms.
        # Some raise ValueError for unknown kwargs (e.g., use_gpu, show_log).
        base_kwargs = {
            "lang": self.lang,
            "use_angle_cls": self.use_angle_cls,
        }

        attempts = [
            base_kwargs,
            {**base_kwargs, "use_gpu": self.use_gpu},
        ]

        last_err: Exception | None = None
        for kwargs in attempts:
            try:
                self._engine = PaddleOCR(**kwargs)
                return
            except (TypeError, ValueError) as e:
                last_err = e
                self._engine = None

        raise RuntimeError(
            "PaddleOCR initialization failed for all attempted constructor signatures.\n"
            f"Last error: {repr(last_err)}\n"
            f"Attempts: {attempts}\n"
        )

    # ---------------------------------------------------------------------
    # Utils
    # ---------------------------------------------------------------------
    @staticmethod
    def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
        """PaddleOCR expects an OpenCV-like BGR uint8 array."""
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        arr = np.asarray(pil)  # RGB uint8
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] != 3:
            arr = arr[:, :, :3]
        return arr[:, :, ::-1].copy()  # BGR

    @staticmethod
    def _parse_classic_ocr_result(result: Any, det: bool) -> str:
        """
        Normalize classic PaddleOCR .ocr output into a single string.
        Common shapes:
          - For one image: result = [inner]
          - det=True: inner = [[box, (text, score)], ...]
          - det=False: inner = [(text, score), ...] (sometimes still det-shape)
        """
        if result is None:
            return ""

        inner = result
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            inner = result[0]

        if not isinstance(inner, list):
            return ""

        texts: List[str] = []

        if det:
            for item in inner:
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rec = item[1]
                        if isinstance(rec, (list, tuple)) and len(rec) >= 1:
                            t = rec[0]
                            if isinstance(t, str) and t.strip():
                                texts.append(t.strip())
                except Exception:
                    continue
            return "\n".join(texts).strip()

        for item in inner:
            if not isinstance(item, (list, tuple)):
                continue

            # rec-only shape: (text, score)
            if len(item) == 2 and isinstance(item[0], str):
                t = item[0].strip()
                if t:
                    texts.append(t)
                continue

            # det-shape fallback
            if len(item) >= 2 and isinstance(item[1], (list, tuple)) and len(item[1]) >= 1:
                t = item[1][0]
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())

        return "\n".join(texts).strip()

    @staticmethod
    def _extract_texts_generic(obj: Any) -> List[str]:
        """
        Extract text strings from arbitrary nested structures returned by predict() pipelines.
        We walk lists/tuples/dicts and collect plausible strings (often under keys like 'text').
        """
        texts: List[str] = []

        def walk(x: Any) -> None:
            if x is None:
                return
            if isinstance(x, str):
                s = x.strip()
                if s:
                    texts.append(s)
                return
            if isinstance(x, dict):
                # common keys
                for k in ("text", "texts", "rec_texts", "rec_text", "label"):
                    if k in x:
                        walk(x[k])
                # also traverse everything
                for v in x.values():
                    walk(v)
                return
            if isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
                return
            # ignore numbers / arrays / objects

        walk(obj)
        return texts

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        """
        Args:
          block_imgs: list of PIL.Image or array-like
          ocr_config:
            - preprocess: "none" | "gray" | "light" (default "none")
            - det: bool (default False)          # used only by classic .ocr API
            - rec: bool (default True)          # used only by classic .ocr API
            - cls: bool (default False)         # used only by classic .ocr API (and only if use_angle_cls=True)
            - fallback_det_if_empty: bool (default True)  # used only by classic .ocr API
        Returns:
          list[str] aligned with block_imgs
        """
        if not block_imgs:
            return []

        self._lazy_load()
        engine = self._engine

        preprocess_mode = ocr_config.get("preprocess", "none")
        det = bool(ocr_config.get("det", False))
        rec = bool(ocr_config.get("rec", True))
        cls = bool(ocr_config.get("cls", False)) and bool(self.use_angle_cls)
        fallback_det_if_empty = bool(ocr_config.get("fallback_det_if_empty", True))

        out: List[str] = []

        for im in block_imgs:
            # Normalize to PIL RGB
            if isinstance(im, Image.Image):
                pil = im.convert("RGB")
            else:
                arr = np.asarray(im)
                if arr.ndim == 2:
                    pil = Image.fromarray(arr).convert("RGB")
                else:
                    pil = Image.fromarray(arr).convert("RGB")

            pil = preprocess_pil(pil, mode=preprocess_mode)
            bgr = self._pil_to_bgr(pil)

            # Robust dispatch:
            # Try classic engine.ocr with det/rec/cls.
            # If it fails because predict() doesn't accept those kwargs, fall back to engine.predict(img).
            txt = ""

            try:
                # Classic path
                res = engine.ocr(bgr, det=det, rec=rec, cls=cls)
                txt = self._parse_classic_ocr_result(res, det=det)

                if (not txt) and (not det) and fallback_det_if_empty:
                    res2 = engine.ocr(bgr, det=True, rec=True, cls=cls)
                    txt = self._parse_classic_ocr_result(res2, det=True)

            except TypeError as e:
                # Newer pipeline path: PaddleOCR.ocr proxies to predict() and predict() rejects det/rec/cls kwargs
                msg = str(e)
                if "unexpected keyword argument" not in msg:
                    raise

                if not hasattr(engine, "predict") or not callable(getattr(engine, "predict")):
                    raise RuntimeError(
                        "PaddleOCR does not support classic .ocr(det/rec/cls) and has no callable .predict()."
                    ) from e

                res = engine.predict(bgr)
                texts = self._extract_texts_generic(res)
                txt = "\n".join(texts).strip()

            out.append(txt)

        return out


    def validate_ocr_config(self, ocr_config: Dict[str, Any]) -> None:
        # PaddleOCR lang est plutôt dans __init__
        if not getattr(self, "lang", None):
            raise ValueError(
                "[PaddleOcrBackend] 'lang' must be provided at init. "
                "Example: PaddleOcrBackend(lang='de')"
            )

        mode = str(ocr_config.get("preprocess", "none")).lower().strip()
        if mode not in {"none", "gray", "light"}:
            raise ValueError(
                f"[PaddleOcrBackend] Invalid ocr_config['preprocess']={mode!r}. Use none|gray|light."
            )

        # det/rec/cls: accept but don't require; ensure types if present
        for k in ("det", "rec", "cls"):
            if k in ocr_config and not isinstance(ocr_config[k], bool):
                raise TypeError(f"[PaddleOcrBackend] ocr_config['{k}'] must be bool if provided.")
