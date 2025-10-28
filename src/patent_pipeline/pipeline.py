# 📄 src/patent_pipeline/pipeline.py
from pathlib import Path
import argparse
from patent_pipeline.patent_ocr.ocr_utils import process_all_docs
from patent_pipeline.pydantic_extraction.hf_agent import batch_extract_features


# ----------------------------------------------------------------------
# 📂 Dossiers par défaut
# ----------------------------------------------------------------------
RAW_DIR = Path("data/raw_pdf")
OCR_DIR = Path("data/ocr_text")
PRED_DIR = Path("data/predictions")
REPORT_FILE = Path("data/ocr_report.csv")

# RAW_DIR = Path("data/gold_standard_DE/PNGs_extracted")
# OCR_DIR = Path("data/gold_standard_DE/ocr_text")
# PRED_DIR = Path("data/gold_standard_DE/predictions")
# REPORT_FILE = Path("data/gold_standard_DE/ocr_report.csv")
# 📄 src/patent_pipeline/pipeline.py

# ----------------------------------------------------------------------
# 🚀 Pipeline principal
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 🚀 Pipeline principale
# ----------------------------------------------------------------------
def run_pipeline(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = OCR_DIR,
    report_file: Path = REPORT_FILE,
    pred_dir: Path = PRED_DIR,
    backend: str = "doctr",
    batch_size: int = 4,
    limit: int = 100,
    force: bool = True,
    country_hint: str = "ch",
):
    """Lance la pipeline complète OCR + extraction LLM."""

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)


    # 1️⃣ OCR parallèle
    process_all_docs(
        raw_dir=raw_dir,
        out_dir=out_dir,
        report_file=report_file,
        country_hint=country_hint,
        force=force,
        backend=backend,
        batch_size=batch_size,
        limit=limit,
    )

    # 2️⃣ Extraction LLM
    batch_extract_features(out_dir, pred_dir / "predictions_all.jsonl", limit=0)

    print("✅ Pipeline completed successfully.")


# ----------------------------------------------------------------------
# 🎛️ Entrypoint CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline OCR + extraction LLM sur corpus de brevets"
    )
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out_dir", type=Path, default=OCR_DIR)
    parser.add_argument("--report_file", type=Path, default=REPORT_FILE)
    parser.add_argument("--backend", type=str, default="doctr")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--country_hint", type=str, default="ch")

    args = parser.parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        report_file=args.report_file,
        backend=args.backend,
        batch_size=args.batch_size,
        limit=args.limit,
        force=args.force,
        country_hint=args.country_hint,
    )
