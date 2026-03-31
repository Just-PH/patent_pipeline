from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from patent_pipeline.patent_ocr.custom_segmentation import CustomSegmentation
from patent_pipeline.patent_ocr.deskewer import Deskewer
from patent_pipeline.patent_ocr.pipeline_modular import PipelineOCRConfig, Pipeline_OCR, write_report_csv
from patent_pipeline.pydantic_extraction.patent_extractor import PatentExtractor

# ----------------------------------------------------------------------
# Default paths
# ----------------------------------------------------------------------
RAW_DIR = Path("data/raw_pdf")
OCR_DIR = Path("data/ocr_text")
PRED_DIR = Path("data/predictions")
REPORT_FILE = Path("data/ocr_report.csv")


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


def _default_backend_spec(backend_key: str) -> str:
    m = {
        "tesseract": "patent_pipeline.patent_ocr.backends.tesseract_backend:TesseractBackend",
        "tesserocr": "patent_pipeline.patent_ocr.backends.tesserocr_backend:TesserocrBackend",
        "doctr": "patent_pipeline.patent_ocr.backends.doctr_backend:DocTROcrBackend",
        "gotocr": "patent_pipeline.patent_ocr.backends.got_ocr_backend:GotOcrBackend",
        "paddle": "patent_pipeline.patent_ocr.backends.paddleocr_backend:PaddleOcrBackend",
        "lightonocr": "patent_pipeline.patent_ocr.backends.lightonocr_backend:LightOnOcrBackend",
    }
    if backend_key not in m:
        raise ValueError(
            f"Unknown backend key '{backend_key}'. "
            f"Use one of {sorted(m)} or pass a full import path via backend_import."
        )
    return m[backend_key]


def _load_prompt_template(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    template = path.read_text(encoding="utf-8")
    if "{text}" not in template:
        raise ValueError("Prompt template must contain {text}")
    return template


def _build_ocr_pipeline(
    *,
    segmentation: str,
    backend_obj: Any,
    deskew: bool,
    deskew_max_angle: float,
    deskew_method: str,
):
    deskewer = Deskewer(method=deskew_method) if deskew else None
    if segmentation == "custom":
        segmenter = CustomSegmentation(deskewer=deskewer, deskew_max_angle=deskew_max_angle)
    else:
        segmenter = None

    return Pipeline_OCR(
        ocr_backend=backend_obj,
        segmenter=segmenter,
        deskewer=deskewer,
    )


def _write_ocr_only_jsonl(
    *,
    rows,
    out_dir: Path,
    jsonl_path: Path,
) -> int:
    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for r in rows:
            if r.status != "ok":
                continue
            txt_path = out_dir / f"{Path(r.file_name).stem}.txt"
            record = {
                "file_name": txt_path.name,
                "ocr_path": str(txt_path),
                "status": "ocr_only",
                "ocr_text": txt_path.read_text(encoding="utf-8"),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def run_pipeline(
    *,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = OCR_DIR,
    report_file: Path = REPORT_FILE,
    pred_dir: Path = PRED_DIR,
    # OCR config
    backend: str = "doctr",
    backend_import: Optional[str] = None,
    backend_kwargs: Optional[Dict[str, Any]] = None,
    ocr_config: Optional[Dict[str, Any]] = None,
    segmentation: str = "custom",
    deskew: bool = True,
    deskew_max_angle: float = 20.0,
    deskew_method: str = "hough",
    workers: int = 1,
    parallel: str = "none",
    limit_ocr: Optional[int] = None,
    force: bool = True,
    keep_empty_docs: bool = True,
    timings_ocr: str = "off",
    # Extraction config
    skip_extraction: bool = False,
    model_name: Optional[str] = None,
    llm_backend: str = "auto",
    device: Optional[str] = None,
    quantization: str = "none",
    attn_implementation: str = "auto",
    cache_implementation: str = "auto",
    prompt_id: Optional[str] = None,
    prompt_template_path: Optional[Path] = None,
    guardrail_profile: str = "auto",
    max_ocr_chars: int = 10000,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    do_sample: bool = False,
    timings_llm: str = "off",
    limit_llm: Optional[int] = None,
) -> None:
    """
    Full pipeline: OCR -> LLM extraction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    backend_kwargs = backend_kwargs or {}
    ocr_config = ocr_config or {}

    raw_path = Path(raw_dir)
    single_doc = raw_path.is_file()

    backend_spec = backend_import or _default_backend_spec(backend)
    BackendCls = _import_symbol(backend_spec)
    backend_obj = BackendCls(**backend_kwargs)

    pipeline = _build_ocr_pipeline(
        segmentation=segmentation,
        backend_obj=backend_obj,
        deskew=deskew,
        deskew_max_angle=deskew_max_angle,
        deskew_method=deskew_method,
    )

    cfg = PipelineOCRConfig(
        raw_dir=raw_path if not single_doc else raw_path.parent,
        out_dir=out_dir,
        report_file=report_file,
        segmentation_mode=segmentation,
        deskew=deskew,
        deskew_max_angle=deskew_max_angle,
        ocr_config=ocr_config,
        workers=workers,
        parallel=parallel,
        limit=limit_ocr,
        force=force,
        keep_empty_docs=keep_empty_docs,
        timings=timings_ocr,
    )

    if single_doc:
        rows = [pipeline._process_one(raw_path, cfg)]
    else:
        rows = pipeline.run(cfg)

    write_report_csv(report_file, rows)

    jsonl_path = pred_dir / "predictions_all.jsonl"
    if skip_extraction:
        written = _write_ocr_only_jsonl(rows=rows, out_dir=out_dir, jsonl_path=jsonl_path)
        print(f"Wrote {written} OCR-only records to {jsonl_path}")
        return

    prompt_template = _load_prompt_template(prompt_template_path)
    extractor = PatentExtractor(
        model_name=model_name,
        backend=llm_backend,
        device=device,
        quantization=quantization,
        attn_implementation=attn_implementation,
        cache_implementation=cache_implementation,
        prompt_id=prompt_id,
        prompt_template=prompt_template,
        guardrail_profile=guardrail_profile,
        max_ocr_chars=max_ocr_chars,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        timings=timings_llm,
    )

    if single_doc:
        txt_path = out_dir / f"{raw_path.stem}.txt"
        record = extractor.extract_from_file(txt_path)
        jsonl_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote 1 prediction to {jsonl_path}")
    else:
        extractor.batch_extract(txt_dir=out_dir, out_file=jsonl_path, limit=limit_llm)
        print(f"Wrote predictions to {jsonl_path}")
