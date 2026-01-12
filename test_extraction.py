# test_extraction.py

from pathlib import Path
from patent_pipeline.pydantic_extraction.patent_extractor import PatentExtractor
from patent_pipeline.pydantic_extraction.prompt_templates import (
    PROMPT_EXTRACTION_V1,  # Version détaillée avec exemples
    PROMPT_EXTRACTION_V2,  # Version concise avec few-shot
    PROMPT_EXTRACTION_V3  # Version ultra-simple
)

def main():


    ocr_text = Path("data/gold_standard_DE/ocr_text_v2/DE-100639-C.txt").read_text(
        encoding="utf-8", errors="replace"
    )

    # Test 1: Version english

    extractor = PatentExtractor(
        model_name="mlx-community/Mistral-7B-Instruct-v0.3",
        backend="mlx",
        prompt_template=PROMPT_EXTRACTION_V2,
        max_new_tokens=1024
    )
    # Traite tout le dossier
    txt_dir = Path("data/gold_standard_DE/ocr_text_v2")
    out_file = Path("output/patent_extractions.jsonl")

    # Lance le batch processing
    count = extractor.batch_extract(
        txt_dir=txt_dir,
        out_file=out_file,
        limit=None  # None = tous les fichiers, ou mets un nombre pour tester
    )
    print(f"\n🎉 Terminé ! {count} brevets traités")
    print(f"📁 Résultats dans : {out_file}")
    # for prompt in [PROMPT_EXTRACTION_V2]:#[PROMPT_EXTRACTION_V1, PROMPT_EXTRACTION_V2, PROMPT_EXTRACTION_V3]:
    #     print(f"--- Testing prompt version {i} ---")
    #     i += 1
    #     extractor.set_prompt_template(prompt)
    #     result = extractor.extract(ocr_text)
    #     for key, value in result.prediction:
    #         print(f"{key}: {value}")
    #     print("\n\n")
    #     print("========================================\n\n")
if __name__ == "__main__":
    main()
