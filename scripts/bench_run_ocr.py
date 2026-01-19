#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

def _to_jsonable(x):
    """Recursively convert objects (Path, dataclasses, etc.) into JSON-serializable types."""
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # basic types are fine; everything else: fallback to string
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _git_info(repo_dir: Path) -> Dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            out = subprocess.check_output(cmd, cwd=str(repo_dir), stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
    }


def _import_symbol(spec: str):
    """
    spec formats:
      - "package.module:ClassName"
      - "package.module.ClassName"
    """
    if ":" in spec:
        mod, name = spec.split(":", 1)
    else:
        parts = spec.split(".")
        if len(parts) < 2:
            raise ValueError(f"Bad import spec: {spec}")
        mod, name = ".".join(parts[:-1]), parts[-1]

    import importlib

    m = importlib.import_module(mod)
    try:
        return getattr(m, name)
    except AttributeError as e:
        raise ImportError(f"Cannot find symbol {name} in module {mod}") from e


def _ensure_src_on_path(repo_dir: Path) -> None:
    src = repo_dir / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _default_backend_spec(backend_key: str) -> str:
    """
    Map short keys -> your backend classes.
    Adjust here if your module/class names differ.
    """
    m = {
        "tesseract": "patent_pipeline.patent_ocr.backends.tesseract_backend:TesseractBackend",
        "doctr": "patent_pipeline.patent_ocr.backends.doctr_backend:DocTROcrBackend",
        "gotocr": "patent_pipeline.patent_ocr.backends.got_ocr_backend:GotOcrBackend",
        "paddle": "patent_pipeline.patent_ocr.backends.paddleocr_backend:PaddleOcrBackend",
    }
    if backend_key not in m:
        raise ValueError(
            f"Unknown backend key '{backend_key}'. "
            f"Use one of {sorted(m)} or pass a full import path via --backend-import."
        )
    return m[backend_key]


def _build_pipeline(segmentation_mode: str, backend_obj, deskew_method: str = "hough"):
    from patent_pipeline.patent_ocr.deskewer import Deskewer
    from patent_pipeline.patent_ocr.custom_segmentation import CustomSegmentation
    from patent_pipeline.patent_ocr.pipeline_modular import Pipeline_OCR

    deskewer = Deskewer(method=deskew_method)

    if segmentation_mode == "custom":
        segmenter = CustomSegmentation(deskewer=deskewer, deskew_max_angle=20.0)
    else:
        segmenter = None

    pipeline = Pipeline_OCR(
        ocr_backend=backend_obj,
        segmenter=segmenter,
        deskewer=deskewer,
    )
    return pipeline, deskewer


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner — OCR stage (writes texts/{identifier}.txt + ocr_report.csv + run.json)"
    )
    parser.add_argument("--raw-dir", type=Path, required=True, help="Input images/PDF directory")
    parser.add_argument("--out-root", type=Path, required=True, help="Root output directory (will create out-root/run-name/)")
    parser.add_argument("--run-name", type=str, required=True, help="Run folder name (e.g. gotocr_custom_v1)")
    parser.add_argument("--segmentation", type=str, default="custom", choices=["custom", "backend"], help="Segmentation mode for PipelineOCRConfig")
    parser.add_argument("--backend", type=str, default="gotocr", help="Backend short key (tesseract/doctr/gotocr/paddle)")
    parser.add_argument("--backend-import", type=str, default=None, help="Override backend import spec: module:Class")
    parser.add_argument("--backend-kwargs-json", type=str, default="{}", help="JSON kwargs passed to backend constructor")
    parser.add_argument("--ocr-config-json", type=str, default="{}", help="JSON dict passed as cfg.ocr_config")
    parser.add_argument("--deskew", action="store_true", help="Enable deskew (default True)")
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskew")
    parser.add_argument("--deskew-max-angle", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing txt outputs")
    parser.add_argument("--keep-empty-docs", action="store_true", help="Write empty txt files too")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel", type=str, default="none", choices=["none", "threads", "processes"])
    parser.add_argument("--deskew-method", type=str, default="hough", help="Deskew method used by Deskewer")
    parser.add_argument(
        "--timings",
        type=str,
        default="off",
        choices=["off", "basic", "detailed"],
        help="Timing measurements: off|basic (total+ocr)|detailed (load/deskew/segment/ocr/write/total)",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]  # patent_pipeline/
    _ensure_src_on_path(repo_dir)

    out_dir = args.out_root / args.run_name
    texts_dir = out_dir / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)

    report_file = out_dir / "ocr_report.csv"
    run_json = out_dir / "run.json"

    # Decide deskew bool
    deskew = True
    if args.no_deskew:
        deskew = False
    elif args.deskew:
        deskew = True

    backend_spec = args.backend_import or _default_backend_spec(args.backend)
    BackendCls = _import_symbol(backend_spec)

    backend_kwargs = json.loads(args.backend_kwargs_json or "{}")
    ocr_config = json.loads(args.ocr_config_json or "{}")

    # Instantiate backend
    try:
        backend_obj = BackendCls(**backend_kwargs)
    except TypeError as e:
        raise RuntimeError(
            f"Failed to instantiate backend {backend_spec} with kwargs {backend_kwargs}. "
            f"Fix --backend-kwargs-json.\nOriginal: {e}"
        ) from e

    pipeline, deskewer = _build_pipeline(args.segmentation, backend_obj, deskew_method=args.deskew_method)

    from patent_pipeline.patent_ocr.pipeline_modular import PipelineOCRConfig, write_report_csv

    cfg = PipelineOCRConfig(
        raw_dir=args.raw_dir,
        out_dir=texts_dir,              # <- writes {stem}.txt inside texts/
        report_file=report_file,
        segmentation_mode=args.segmentation,
        deskew=deskew,
        deskew_max_angle=args.deskew_max_angle,
        ocr_config=ocr_config,
        workers=args.workers,
        parallel=args.parallel,
        limit=args.limit,
        force=args.force,
        keep_empty_docs=args.keep_empty_docs,
        timings=args.timings,
    )

    # Save run.json (metadata + full config)
    meta = {
        "kind": "ocr",
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "raw_dir": str(args.raw_dir),
        "out_dir": str(out_dir),
        "texts_dir": str(texts_dir),
        "report_file": str(report_file),
        "segmentation": args.segmentation,
        "timings": args.timings,
        "backend_spec": backend_spec,
        "backend_kwargs": backend_kwargs,
        "ocr_config": ocr_config,
        "pipeline_cfg": _to_jsonable(cfg),
        **_git_info(repo_dir),
        "env": {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
    }
    run_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Run
    rows = pipeline.run(cfg)
    write_report_csv(report_file, rows)

    print("✅ OCR done")
    print("Run dir:", out_dir)
    print("Texts dir:", texts_dir)
    print("Report:", report_file)
    print("Run meta:", run_json)


if __name__ == "__main__":
    main()
