from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image
import pytesseract

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patent_pipeline.patent_ocr.utils.image_preprocess import preprocess_pil

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_PREPROCESS = "light"
_LANG = "frk+deu"
_CONFIG = "--psm 3 --oem 3 -c preserve_interword_spaces=1"
_TESSEROCR_API = None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _init_common(preprocess: str, lang: str, config: str) -> None:
    global _PREPROCESS, _LANG, _CONFIG
    _PREPROCESS = preprocess
    _LANG = lang
    _CONFIG = config


def _init_pytesseract(preprocess: str, lang: str, config: str) -> None:
    _init_common(preprocess, lang, config)


def _init_tesserocr(preprocess: str, lang: str, config: str, tessdata: str) -> None:
    _init_common(preprocess, lang, config)
    global _TESSEROCR_API
    import tesserocr

    _TESSEROCR_API = tesserocr.PyTessBaseAPI(
        path=tessdata,
        lang=lang,
        psm=tesserocr.PSM.AUTO,     # psm 3
        oem=tesserocr.OEM.DEFAULT,  # proche oem 3
    )
    if "preserve_interword_spaces=1" in config:
        _TESSEROCR_API.SetVariable("preserve_interword_spaces", "1")


def _ocr_pytesseract(path_str: str) -> str:
    img = preprocess_pil(Image.open(path_str).convert("RGB"), mode=_PREPROCESS)
    return (pytesseract.image_to_string(img, lang=_LANG, config=_CONFIG) or "").strip()


def _ocr_tesserocr(path_str: str) -> str:
    global _TESSEROCR_API
    if _TESSEROCR_API is None:
        raise RuntimeError("tesserocr API not initialized in worker")
    img = preprocess_pil(Image.open(path_str).convert("RGB"), mode=_PREPROCESS)
    _TESSEROCR_API.SetImage(img)
    return (_TESSEROCR_API.GetUTF8Text() or "").strip()


def _run_pool(
    paths: list[Path],
    fn,
    *,
    workers: int,
    parallel: str,
    initializer=None,
    initargs=(),
) -> tuple[list[str], float]:
    ex_cls = ProcessPoolExecutor if parallel == "processes" else ThreadPoolExecutor
    t0 = time.perf_counter()
    with ex_cls(max_workers=workers, initializer=initializer, initargs=initargs) as ex:
        texts = list(ex.map(fn, [str(p) for p in paths]))
    return texts, time.perf_counter() - t0


def _auto_tessdata(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"tessdata path not found: {p}")

    candidates = [
        Path("/opt/homebrew/share/tessdata"),
        Path("/usr/local/share/tessdata"),
        Path("/usr/share/tessdata"),
    ]
    for p in candidates:
        if (p / "deu.traineddata").exists() and (p / "frk.traineddata").exists():
            return p
    raise RuntimeError("Cannot find tessdata with deu+frk traineddata")


def _iter_images(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("data/gold_standard_DE/PNGs_extracted"))
    ap.add_argument("--out-root", type=Path, default=Path("output/compare_tesseract_wrappers"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--parallel", type=str, choices=["threads", "processes"], default="processes")
    ap.add_argument("--preprocess", type=str, choices=["none", "gray", "light"], default="light")
    ap.add_argument("--lang", type=str, default="frk+deu")
    ap.add_argument("--config", type=str, default="--psm 3 --oem 3 -c preserve_interword_spaces=1")
    ap.add_argument("--tessdata", type=str, default=None)
    args = ap.parse_args()

    files = list(_iter_images(args.input_dir))[: args.limit]
    if not files:
        raise SystemExit(f"No images found in {args.input_dir}")

    tessdata = _auto_tessdata(args.tessdata)
    out_py = args.out_root / "pytesseract_txt"
    out_to = args.out_root / "tesserocr_txt"
    out_py.mkdir(parents=True, exist_ok=True)
    out_to.mkdir(parents=True, exist_ok=True)

    print(f"files={len(files)} workers={args.workers} parallel={args.parallel}")
    print(f"input_dir={args.input_dir}")
    print(f"tessdata={tessdata}")

    py_texts, py_s = _run_pool(
        files,
        _ocr_pytesseract,
        workers=args.workers,
        parallel=args.parallel,
        initializer=_init_pytesseract,
        initargs=(args.preprocess, args.lang, args.config),
    )
    to_texts, to_s = _run_pool(
        files,
        _ocr_tesserocr,
        workers=args.workers,
        parallel=args.parallel,
        initializer=_init_tesserocr,
        initargs=(args.preprocess, args.lang, args.config, str(tessdata)),
    )

    rows = []
    exact = 0
    norm_eq = 0
    for p, py_t, to_t in zip(files, py_texts, to_texts):
        (out_py / f"{p.name}.txt").write_text(py_t, encoding="utf-8")
        (out_to / f"{p.name}.txt").write_text(to_t, encoding="utf-8")

        e = py_t == to_t
        n = _norm(py_t) == _norm(to_t)
        if e:
            exact += 1
        if n:
            norm_eq += 1

        ratio = difflib.SequenceMatcher(None, _norm(py_t), _norm(to_t)).ratio()
        rows.append(
            {
                "file": p.name,
                "exact_equal": int(e),
                "normalized_equal": int(n),
                "similarity_ratio_norm": f"{ratio:.6f}",
                "len_pytesseract": len(py_t),
                "len_tesserocr": len(to_t),
            }
        )

    cmp_csv = args.out_root / "comparison.csv"
    with open(cmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "exact_equal",
                "normalized_equal",
                "similarity_ratio_norm",
                "len_pytesseract",
                "len_tesserocr",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    summary = {
        "files": len(files),
        "workers": args.workers,
        "parallel": args.parallel,
        "lang": args.lang,
        "config": args.config,
        "preprocess": args.preprocess,
        "tessdata": str(tessdata),
        "pytesseract_total_s": round(py_s, 3),
        "pytesseract_per_doc_s": round(py_s / len(files), 3),
        "tesserocr_total_s": round(to_s, 3),
        "tesserocr_per_doc_s": round(to_s / len(files), 3),
        "exact_equal": f"{exact}/{len(files)}",
        "normalized_equal": f"{norm_eq}/{len(files)}",
        "out_pytesseract_txt": str(out_py),
        "out_tesserocr_txt": str(out_to),
        "comparison_csv": str(cmp_csv),
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
