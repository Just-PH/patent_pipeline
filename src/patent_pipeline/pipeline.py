# 📄 src/patent_pipeline/pipeline.py
from pathlib import Path
import os
from patent_pipeline.patent_ocr.ocr_utils import process_all_docs
from patent_pipeline.pydantic.hf_agent import batch_extract_features
from patent_pipeline.utils.device_utils import print_device_info

RAW_DIR = Path("data/raw_pdf")
OCR_DIR = Path("data/ocr_text")
PRED_DIR = Path("data/predictions")
REPORT_FILE = Path("data/ocr_report.csv")

def run_pipeline():
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # 1️⃣ OCR parallèle
    process_all_docs(
        raw_dir=RAW_DIR,
        out_dir=OCR_DIR,
        report_file=REPORT_FILE,
        country_hint="ch",
        force=False,
        threads=4,
        backend='doctr',
        limit = None
    )

    # 2️⃣ Extraction LLM
    batch_extract_features(OCR_DIR, PRED_DIR / "predictions_all.jsonl",limit=None)


if __name__ == "__main__":
    run_pipeline()
