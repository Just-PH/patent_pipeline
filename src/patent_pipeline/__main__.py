# src/patent_pipeline/__main__.py
import argparse
from pathlib import Path
from patent_pipeline.pipeline import run_pipeline
# ----------------------------------------------------------------------
# 📂 Dossiers par défaut
# ----------------------------------------------------------------------
# RAW_DIR = Path("data/raw_pdf")
# OCR_DIR = Path("data/ocr_text")
# PRED_DIR = Path("data/predictions")
# REPORT_FILE = Path("data/ocr_report.csv")

RAW_DIR = Path("data/gold_standard_DE/PNGs_extracted")
OCR_DIR = Path("data/gold_standard_DE/ocr_text")
PRED_DIR = Path("data/gold_standard_DE/predictions")
REPORT_FILE = Path("data/gold_standard_DE/ocr_report.csv")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline OCR + extraction LLM sur corpus de brevets"
    )
    parser.add_argument("--raw_dir", type=str, default=RAW_DIR)
    parser.add_argument("--out_dir", type=str, default=OCR_DIR)
    parser.add_argument("--pred_dir", type=str, default=PRED_DIR)
    parser.add_argument("--report_file", type=str, default=REPORT_FILE)
    parser.add_argument("--backend", type=str, default="doctr", choices=["tesseract", "doctr", "easyocr"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--limit_ocr", type=int, default=None)
    parser.add_argument("--limit_llm", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--country_hint", type=str, default="de")
    parser.add_argument("--preproc_method", type=str, default="sauvola")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip LLM feature extraction (only OCR).")


    args = parser.parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        pred_dir=args.pred_dir,
        report_file=args.report_file,
        backend=args.backend,
        batch_size=args.batch_size,
        threads=args.threads,
        limit_ocr=args.limit_ocr,
        limit_llm=args.limit_llm,
        force=args.force,
        country_hint=args.country_hint,
        preproc_method=args.preproc_method,
        skip_extraction = args.skip_extraction,
    )

if __name__ == "__main__":
    main()
