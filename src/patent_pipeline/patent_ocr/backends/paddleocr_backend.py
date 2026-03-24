# src/patent_pipeline/patent_ocr/backends/paddleocr_backend.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from patent_pipeline.patent_ocr.utils.image_preprocess import preprocess_pil


@dataclass
class PaddleOcrBackend:
    """
    PaddleOCR backend aligned with your DocTR/Tesseract backend interface.

    Goals:
      - Be robust to PaddleOCR API differences across versions.
      - CPU-only (no Paddle GPU logic).
      - Lazy import so your project can run without PaddleOCR installed (until used).

    Notes on PaddleOCR APIs (version-dependent):
      - Classic: engine.ocr(img, det=..., rec=..., cls=...)
      - Newer pipelines: .ocr(...) may proxy to .predict(...) and reject det/rec/cls kwargs.

    This backend:
      - Tries engine.ocr(..., det/rec/cls)
      - If it fails with "unexpected keyword argument", falls back to engine.predict(img)
    """

    # Engine configuration
    lang: str = "en"
    use_angle_cls: bool = False

    # CPU-only
    device: str = "cpu"

    # Exposed name
    name_: str = "paddleocr"

    # Internals
    _engine: Any = field(init=False, default=None, repr=False)
    _resolved_device: str = field(init=False, default="cpu", repr=False)
    _use_gpu: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.lang = (self.lang or "en").lower().strip()
        self.device = "cpu"

    @property
    def name(self) -> str:
        return self.name_

    @property
    def is_gpu(self) -> bool:
        # Pipeline uses this to avoid naive multiprocessing for GPU backends.
        return bool(self._use_gpu)

    # ---------------------------------------------------------------------
    # Lazy init + imports
    # ---------------------------------------------------------------------
    @staticmethod
    def _import_paddleocr() -> Any:
        """
        Import PaddleOCR in the most compatible way.
        Raises a clear error if missing.
        """
        try:
            from paddleocr import PaddleOCR  # type: ignore
            return PaddleOCR
        except Exception as e:
            raise RuntimeError(
                "Missing deps for PaddleOCR backend.\n"
                "Install inside the SAME interpreter/env as your runner:\n"
                "  python -m pip install -U paddlepaddle paddleocr\n"
                "\n"
                "If you want GPU (CUDA) in Linux:\n"
                "  - you must install a PaddlePaddle build compiled with CUDA\n"
                "  - and have a working NVIDIA driver\n"
            ) from e

    def _resolve_device(self) -> None:
        """
        Resolve requested device into:
          - self._resolved_device in {"cuda","cpu"}
          - self._use_gpu bool
        """
        self._resolved_device = "cpu"
        self._use_gpu = False

    def _lazy_load(self, extra_init_args: Optional[Dict[str, Any]] = None) -> None:
        """
        Load engine if not loaded.
        extra_init_args: params to inject into PaddleOCR constructor (e.g. det_db_thresh).
        """
        if self._engine is not None:
            return

        PaddleOCR = self._import_paddleocr()
        self._resolve_device()

        # PaddleOCR constructor kwargs vary across versions/platforms.
        # We'll try a few signatures in a safe order.
        base_kwargs = {
            "lang": self.lang,
            "use_angle_cls": self.use_angle_cls,
        }

        # PATCH: Inject user overrides (paddle_init from config)
        if extra_init_args:
            base_kwargs.update(extra_init_args)

        attempts: List[Dict[str, Any]] = [
            {**base_kwargs, "use_gpu": False, "show_log": False},
            {**base_kwargs, "use_gpu": False},
            base_kwargs,
        ]

        last_err: Optional[Exception] = None
        for kwargs in attempts:
            try:
                self._engine = PaddleOCR(**kwargs)
                return
            except (TypeError, ValueError) as e:
                last_err = e
                self._engine = None

        raise RuntimeError(
            "PaddleOCR initialization failed for all attempted constructor signatures.\n"
            f"Requested device={self.device!r} resolved_device={self._resolved_device!r} use_gpu={self._use_gpu}\n"
            f"Last error: {repr(last_err)}\n"
            f"Attempts: {attempts}\n"
        )

    # ---------------------------------------------------------------------
    # Patch minimal: allow lang override via ocr_config (and reload engine)
    # ---------------------------------------------------------------------
    def _maybe_override_lang_from_ocr_config(self, ocr_config: Dict[str, Any]) -> None:
        """
        If ocr_config contains 'lang', override self.lang for this run.
        """
        if not ocr_config:
            return
        raw = ocr_config.get("lang", None)
        if raw is None:
            return
        new_lang = str(raw).lower().strip()
        if not new_lang:
            return
        if new_lang != self.lang:
            # IMPORTANT:
            #  - PaddleOCR downloads/loads models during PaddleOCR(...) init, based on lang.
            #  - Therefore changing lang MUST force a re-init BEFORE any .ocr() call,
            #    otherwise we'd keep using the previously loaded models.
            self.lang = new_lang
            self._engine = None  # force re-init with the new lang (and thus download if missing)

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
                for k in ("text", "texts", "rec_texts", "rec_text", "label"):
                    if k in x:
                        walk(x[k])
                for v in x.values():
                    walk(v)
                return
            if isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
                return

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
            - lang: str (optional)
            - preprocess: "none" | "gray" | "light"
            - paddle_init: dict (optional) -> Passed to PaddleOCR constructor (e.g. det_db_thresh)
            - det, rec, cls: bool
        """
        if not block_imgs:
            return []

        # PATCH: Update lang if needed
        self._maybe_override_lang_from_ocr_config(ocr_config)

        # Optional but safe: if config explicitly toggles angle cls, reload (it affects constructor)
        # We keep your existing design: anything constructor-level must reload.
        if "use_angle_cls" in ocr_config and bool(ocr_config["use_angle_cls"]) != bool(self.use_angle_cls):
            self.use_angle_cls = bool(ocr_config["use_angle_cls"])
            self._engine = None

        # Extract paddle specific init params from config
        paddle_init_params = ocr_config.get("paddle_init", None)

        if paddle_init_params:
            self._engine = None

        self._lazy_load(extra_init_args=paddle_init_params)
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
                pil = Image.fromarray(arr).convert("RGB")

            pil = preprocess_pil(pil, mode=preprocess_mode)
            bgr = self._pil_to_bgr(pil)

            txt = ""

            try:
                # Classic path
                res = engine.ocr(bgr, det=det, rec=rec, cls=cls)
                txt = self._parse_classic_ocr_result(res, det=det)

                if (not txt) and (not det) and fallback_det_if_empty:
                    res2 = engine.ocr(bgr, det=True, rec=True, cls=cls)
                    txt = self._parse_classic_ocr_result(res2, det=True)

            except TypeError as e:
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
        if "lang" in ocr_config:
            lang = str(ocr_config.get("lang") or "").lower().strip()
            if not lang:
                raise ValueError(
                    "[PaddleOcrBackend] Invalid ocr_config['lang']. Provide a non-empty string."
                )

        if not getattr(self, "lang", None):
            raise ValueError(
                "[PaddleOcrBackend] 'lang' must be provided at init. Example: PaddleOcrBackend(lang='de')"
            )

        mode = str(ocr_config.get("preprocess", "none")).lower().strip()
        if mode not in {"none", "gray", "light"}:
            raise ValueError(
                f"[PaddleOcrBackend] Invalid ocr_config['preprocess']={mode!r}. Use none|gray|light."
            )

        for k in ("det", "rec", "cls"):
            if k in ocr_config and not isinstance(ocr_config[k], bool):
                raise TypeError(f"[PaddleOcrBackend] ocr_config['{k}'] must be bool if provided.")
