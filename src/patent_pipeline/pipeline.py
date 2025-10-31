# 📄 src/patent_pipeline/pipeline.py
from pathlib import Path
import json
import csv
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
    threads: int = 3,
    limit_ocr: int = None,
    limit_llm: int = None,
    force: bool = True,
    country_hint: str = "ch",
    preproc_method: str = "sauvola",
    skip_extraction: bool = False,
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
        limit_ocr=limit_ocr,
        threads=threads,
        preproc_method=preproc_method
    )

    # 2️⃣ Extraction LLM
    if not skip_extraction:
        batch_extract_features(out_dir, pred_dir / "predictions_all.jsonl", limit_llm=limit_llm)
    else:
        # Si on skip, on écrit un fichier JSONL minimal avec juste les OCR valides
        jsonl_path = pred_dir / "predictions_all.jsonl"
        report_path = report_file  # CSV généré par l’OCR
        print(f"⚙️ Skipping LLM extraction → writing raw OCR texts to {jsonl_path}")

        # 🧩 Lire la liste des fichiers valides depuis ocr_report.csv
        valid_stems = set()
        if report_path.exists():
            import csv
            with open(report_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                valid_stems = {
                    Path(row["file_name"]).stem
                    for row in reader
                    if row.get("status", "").lower() == "ok"
                }
            print(f"📄 {len(valid_stems)} valid entries found in {report_path.name}")
        else:
            print(f"⚠️ No report file found at {report_path}, using all OCR files.")
            valid_stems = {p.stem for p in out_dir.glob("*.txt")}

        # 🧠 Écrire uniquement les fichiers OCR (.txt) correspondant à ces stems
        written = 0
        with open(jsonl_path, "w", encoding="utf-8") as f_out:
            for txt_file in sorted(out_dir.glob("*.txt")):
                if txt_file.stem not in valid_stems:
                    continue
                record = {
                    "file_name": txt_file.name,
                    "ocr_path": str(txt_file),
                    "status": "ocr_only",
                    "ocr_text": txt_file.read_text(encoding="utf-8")
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1


        print(f"✅ Raw OCR export → {written} documents enregistrés (from report)")



    print("✅ Pipeline completed successfully.")


# # ----------------------------------------------------------------------
# # 🎛️ Entrypoint CLI
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Pipeline OCR + extraction LLM sur corpus de brevets"
#     )
#     parser.add_argument("--raw_dir", type=Path, default=RAW_DIR)
#     parser.add_argument("--out_dir", type=Path, default=OCR_DIR)
#     parser.add_argument("--report_file", type=Path, default=REPORT_FILE)
#     parser.add_argument("--backend", type=str, default="doctr")
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--threads", type=int, default=2)
#     parser.add_argument("--limit", type=int, default=None)
#     parser.add_argument("--force", action="store_true")
#     parser.add_argument("--country_hint", type=str, default="ch")
#     parser.add_argument("--preproc", type=str, default="sauvola")

#     args = parser.parse_args()

#     run_pipeline(
#         raw_dir=args.raw_dir,
#         out_dir=args.out_dir,
#         report_file=args.report_file,
#         backend=args.backend,
#         batch_size=args.batch_size,
#         limit=None,
#         force=args.force,
#         country_hint=args.country_hint,
#         preproc_method=args.preproc
#       )
