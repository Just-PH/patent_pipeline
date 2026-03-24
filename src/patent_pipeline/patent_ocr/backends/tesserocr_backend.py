# src/patent_pipeline/patent_ocr/backends/tesserocr_backend.py
from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
import threading
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from patent_pipeline.patent_ocr.utils.image_preprocess import preprocess_pil


_API_CACHE_LOCK = threading.Lock()
_API_CACHE: Dict[Tuple[int, int, str, int, Optional[str]], Any] = {}


def _parse_kv(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Expected key=value after -c, got: {raw!r}")
    k, v = raw.split("=", 1)
    k = k.strip()
    v = v.strip()
    if not k:
        raise ValueError(f"Invalid empty key in -c expression: {raw!r}")
    return k, v


def _parse_tesseract_config(config: str) -> Tuple[int, int, Dict[str, str]]:
    """
    Parse a Tesseract CLI config string into API-compatible pieces.
    Supported:
      - --psm N | --psm=N
      - --oem N | --oem=N
      - -c key=value (repeatable)
    """
    tokens = shlex.split(config or "")
    psm = 3
    oem = 3
    variables: Dict[str, str] = {}
    unsupported: List[str] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--psm":
            i += 1
            if i >= len(tokens):
                raise ValueError("Missing value after --psm")
            psm = int(tokens[i])
        elif tok.startswith("--psm="):
            psm = int(tok.split("=", 1)[1])
        elif tok == "--oem":
            i += 1
            if i >= len(tokens):
                raise ValueError("Missing value after --oem")
            oem = int(tokens[i])
        elif tok.startswith("--oem="):
            oem = int(tok.split("=", 1)[1])
        elif tok == "-c":
            i += 1
            if i >= len(tokens):
                raise ValueError("Missing key=value after -c")
            k, v = _parse_kv(tokens[i])
            variables[k] = v
        elif tok.startswith("-c") and tok != "-c":
            # Supports -ckey=value form.
            k, v = _parse_kv(tok[2:])
            variables[k] = v
        else:
            unsupported.append(tok)
        i += 1

    if unsupported:
        raise ValueError(
            "[TesserocrBackend] Unsupported tokens in ocr_config['config'] for tesserocr API: "
            f"{unsupported}. Supported: --psm, --oem, -c key=value."
        )

    return psm, oem, variables


@dataclass
class TesserocrBackend:
    """
    Tesseract OCR backend via tesserocr (persistent API handle).

    Notes:
      - CPU backend.
      - Keeps one PyTessBaseAPI per (pid, thread, lang, oem, tessdata_path).
      - Supports the most common config flags used in this repo:
        --psm, --oem, -c key=value.
    """

    name_: str = "tesserocr"
    tessdata_path: Optional[str] = None

    def __post_init__(self) -> None:
        # Import once on the main thread. The tesserocr wheel pulls cysignals,
        # which installs signal handlers and fails if the first import happens
        # from a worker thread.
        self._import_tesserocr()

    @property
    def is_gpu(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self.name_

    @staticmethod
    def _import_tesserocr():
        try:
            import tesserocr  # type: ignore

            return tesserocr
        except Exception as e:
            raise RuntimeError(
                "Missing deps for TesserocrBackend.\n"
                "Install inside the SAME interpreter/env as your runner:\n"
                "  python -m pip install -U tesserocr\n"
                "\n"
                "If tessdata is not auto-detected, pass backend kwargs:\n"
                "  --backend-kwargs-json '{\"tessdata_path\":\"/opt/homebrew/share/tessdata\"}'\n"
                "\n"
                f"Original import error: {repr(e)}\n"
            ) from e

    def _resolve_tessdata_path(self) -> Optional[str]:
        return self.tessdata_path or os.environ.get("TESSDATA_PREFIX")

    def _new_api(self, *, lang: str, oem: int):
        tesserocr = self._import_tesserocr()

        base_kwargs: Dict[str, Any] = {"lang": lang}
        tessdata = self._resolve_tessdata_path()
        if tessdata:
            base_kwargs["path"] = tessdata

        attempts = [
            {**base_kwargs, "oem": oem},
            base_kwargs,
        ]
        last_err: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return tesserocr.PyTessBaseAPI(**kwargs)
            except Exception as e:  # noqa: BLE001
                last_err = e

        raise RuntimeError(
            "Failed to initialize tesserocr.PyTessBaseAPI.\n"
            f"lang={lang!r} oem={oem!r} tessdata_path={tessdata!r}\n"
            f"Last error: {repr(last_err)}"
        )

    def _get_api(self, *, lang: str, oem: int):
        key = (
            os.getpid(),
            threading.get_ident(),
            lang,
            oem,
            self._resolve_tessdata_path(),
        )
        with _API_CACHE_LOCK:
            api = _API_CACHE.get(key)
            if api is None:
                api = self._new_api(lang=lang, oem=oem)
                _API_CACHE[key] = api
        return api

    def run_blocks_ocr(self, block_imgs: List[Any], ocr_config: Dict[str, Any]) -> List[str]:
        if not block_imgs:
            return []

        lang = str(ocr_config.get("lang", "")).strip()
        config = str(ocr_config.get("config", "")).strip()
        preprocess_mode = ocr_config.get("preprocess", "none")

        psm, oem, variables = _parse_tesseract_config(config)
        api = self._get_api(lang=lang, oem=oem)
        api.SetPageSegMode(psm)
        for k, v in variables.items():
            api.SetVariable(k, v)

        out: List[str] = []
        for im in block_imgs:
            pil = im if isinstance(im, Image.Image) else Image.fromarray(im).convert("RGB")
            pil = preprocess_pil(pil, mode=preprocess_mode)
            api.SetImage(pil)
            txt = api.GetUTF8Text()
            out.append((txt or "").strip())

        if len(out) != len(block_imgs):
            raise RuntimeError(f"Tesserocr backend returned {len(out)} texts for {len(block_imgs)} images")
        return out

    def validate_ocr_config(self, ocr_config: Dict[str, Any]) -> None:
        lang = ocr_config.get("lang")
        config = ocr_config.get("config")

        missing = []
        if not lang:
            missing.append("lang")
        if config is None:
            missing.append("config")

        if missing:
            raise ValueError(
                "[TesserocrBackend] Missing required ocr_config keys: "
                f"{missing}. Example:\n"
                "  ocr_config={"
                "\"lang\":\"frk+deu\","
                "\"config\":\"--psm 3 --oem 3 -c preserve_interword_spaces=1\","
                "\"preprocess\":\"light\"}"
            )

        if not isinstance(lang, str):
            raise TypeError("[TesserocrBackend] ocr_config['lang'] must be a string.")
        if not isinstance(config, str):
            raise TypeError("[TesserocrBackend] ocr_config['config'] must be a string.")

        try:
            _parse_tesseract_config(config)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"[TesserocrBackend] Invalid ocr_config['config']: {e}") from e
