import json
from pathlib import Path
from .utils.device_utils import print_device_info
from .patent_ocr.ocr_utils import doc_to_text
from .pydantic.hf_agent import extract_metadata

def main():
    print_device_info()
    pdf = Path("data/raw_pdf/test.pdf")
    ocr = doc_to_text(pdf, lang="fra", is_pdf=True)
    result = extract_metadata(ocr)
    out = Path("data/predictions") / f"{pdf.stem}.jsonl"
    out.write_text(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")
    print(f"✅ Saved: {out}")

if __name__ == "__main__":
    main()
