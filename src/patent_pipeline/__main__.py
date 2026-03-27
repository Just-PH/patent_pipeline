# src/patent_pipeline/__main__.py
import argparse
import json
from pathlib import Path

from patent_pipeline.pipeline import run_pipeline

# ----------------------------------------------------------------------
# Default paths
# ----------------------------------------------------------------------
RAW_DIR = Path("data/raw_pdf")
OCR_DIR = Path("data/ocr_text")
PRED_DIR = Path("data/predictions")
REPORT_FILE = Path("data/ocr_report.csv")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline OCR + extraction LLM sur corpus de brevets"
    )
    # IO
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR, help="Input directory (or single PDF/image file)")
    parser.add_argument("--out_dir", type=Path, default=OCR_DIR, help="OCR texts output directory")
    parser.add_argument("--pred_dir", type=Path, default=PRED_DIR, help="Predictions output directory")
    parser.add_argument("--report_file", type=Path, default=REPORT_FILE, help="OCR report CSV")

    # OCR
    parser.add_argument(
        "--backend",
        type=str,
        default="doctr",
        choices=["tesseract", "tesserocr", "doctr", "gotocr", "paddle", "lightonocr"],
        help="OCR backend key",
    )
    parser.add_argument("--backend_import", type=str, default=None, help="Override backend import path module:Class")
    parser.add_argument("--backend_kwargs_json", type=str, default="{}", help="JSON kwargs for backend constructor")
    parser.add_argument("--ocr_config_json", type=str, default="{}", help="JSON config passed to backend.run_blocks_ocr")
    parser.add_argument("--segmentation", type=str, default="custom", choices=["custom", "backend"])
    parser.add_argument("--deskew", action="store_true", help="Enable deskew")
    parser.add_argument("--no_deskew", action="store_true", help="Disable deskew")
    parser.add_argument("--deskew_max_angle", type=float, default=20.0)
    parser.add_argument("--deskew_method", type=str, default="hough")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--parallel", type=str, default="none", choices=["none", "threads", "processes"])
    parser.add_argument("--limit_ocr", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep_empty_docs", action="store_true")
    parser.add_argument("--timings_ocr", type=str, default="off", choices=["off", "basic", "detailed"])

    # LLM extraction
    parser.add_argument("--skip_extraction", action="store_true", help="Skip LLM extraction (only OCR).")
    parser.add_argument("--model_name", type=str, default=None, help="HF model or local path")
    parser.add_argument("--llm_backend", type=str, default="auto", choices=["auto", "mlx", "pytorch"])
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/mps for pytorch backend")
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "bnb_8bit", "bnb_4bit"],
        help="Optional model quantization for pytorch backend.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "flash_attention_2"],
        help="Attention backend for pytorch backend.",
    )
    parser.add_argument(
        "--cache_implementation",
        type=str,
        default="auto",
        choices=["auto", "dynamic", "static", "offloaded", "offloaded_static"],
        help="KV cache strategy for generate() on pytorch backend.",
    )
    parser.add_argument("--prompt_id", type=str, default=None, help="Prompt id: v1/v2/v3")
    parser.add_argument("--prompt_template_path", type=Path, default=None, help="Path to prompt template containing {text}")
    parser.add_argument("--max_ocr_chars", type=int, default=10000)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--timings_llm", type=str, default="off", choices=["off", "basic", "detailed"])
    parser.add_argument("--limit_llm", type=int, default=None)


    args = parser.parse_args()

    backend_kwargs = json.loads(args.backend_kwargs_json or "{}")
    ocr_config = json.loads(args.ocr_config_json or "{}")

    deskew = True
    if args.no_deskew:
        deskew = False
    elif args.deskew:
        deskew = True

    run_pipeline(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        pred_dir=args.pred_dir,
        report_file=args.report_file,
        backend=args.backend,
        backend_import=args.backend_import,
        backend_kwargs=backend_kwargs,
        ocr_config=ocr_config,
        segmentation=args.segmentation,
        deskew=deskew,
        deskew_max_angle=args.deskew_max_angle,
        deskew_method=args.deskew_method,
        workers=args.workers,
        parallel=args.parallel,
        limit_ocr=args.limit_ocr,
        force=args.force,
        keep_empty_docs=args.keep_empty_docs,
        timings_ocr=args.timings_ocr,
        skip_extraction=args.skip_extraction,
        model_name=args.model_name,
        llm_backend=args.llm_backend,
        device=args.device,
        quantization=args.quantization,
        attn_implementation=args.attn_implementation,
        cache_implementation=args.cache_implementation,
        prompt_id=args.prompt_id,
        prompt_template_path=args.prompt_template_path,
        max_ocr_chars=args.max_ocr_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        timings_llm=args.timings_llm,
        limit_llm=args.limit_llm,
    )

if __name__ == "__main__":
    main()
