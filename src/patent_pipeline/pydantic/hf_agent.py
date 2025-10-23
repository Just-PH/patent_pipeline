# 📄 src/patent_pipeline/pydantic/hf_agent.py
from pathlib import Path
import os
import json
import regex as re
from pydantic import ValidationError
from .models import PatentExtraction, PatentMetadata
from .prompt_templates import PROMPT_EXTRACTION
from ..utils.device_utils import get_device
from tqdm import tqdm

# Optional deps
try:
    import mlx_lm
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -----------------------------------------------------------------------------
HF_MODEL = os.getenv("HF_MODEL", "microsoft/Phi-3-mini-128k-instruct")

_model = None
_tokenizer = None
_pipe = None
_USE_MLX = False

# -----------------------------------------------------------------------------
def _extract_json(text: str) -> str:
    m = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
    return m.group(0) if m else "{}"

# -----------------------------------------------------------------------------
def _load_model():
    """Charge le modèle : MLX (si dispo), sinon CPU PyTorch."""
    global _model, _tokenizer, _pipe, _USE_MLX

    if _pipe is not None or _model is not None:
        return _pipe

    device = get_device()
    print(f"🧠 Using device: {device}")
    print(f"→ Model: {HF_MODEL}")

    if _HAS_MLX:
        print("⚙️  Loading via MLX (quantized int4/int8)")
        _USE_MLX = True
        _model, _tokenizer = mlx_lm.load(HF_MODEL)

        return None  # no pipeline for MLX

    # -------------------------------------------------------------------------
    # PyTorch fallback
    if device == "mps":
        print("⚠️  MPS backend instable → fallback CPU")
        device = "cpu"

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    dtype = torch.float16 if device == "cuda" else torch.float32
    map_arg = device if device == "cuda" else "cpu"

    print(f"🚀 Loading {HF_MODEL} on {device} ({dtype})")
    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    _model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=dtype,
        device_map=map_arg
    )
    _pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_new_tokens=512,
        temperature=0,
        do_sample=False
    )
    return _pipe

# -----------------------------------------------------------------------------
def extract_metadata(ocr_text: str) -> PatentExtraction:
    """OCR → Prompt → JSON → Pydantic."""
    pipe = _load_model()
    prompt = PROMPT_EXTRACTION.format(text=ocr_text)

    if _USE_MLX:
        output = mlx_lm.generate(_model, _tokenizer, prompt, max_tokens=512)
        txt = output.strip()
    else:
        out = pipe(prompt)[0]
        txt = out.get("generated_text") or out.get("text") or ""

    js = _extract_json(txt)
    try:
        data = json.loads(js)
        meta = PatentMetadata(**data)
    except (json.JSONDecodeError, ValidationError):
        meta = PatentMetadata(identifier="unknown")

    return PatentExtraction(ocr_text=ocr_text, model=HF_MODEL, prediction=meta)

# -----------------------------------------------------------------------------
def extract_features_from_txt(txt_path: Path) -> dict:
    """
    Effectue l'extraction complète (OCR déjà fait) pour un seul .txt.
    Retourne un dictionnaire prêt à être sérialisé en JSONL.
    """
    try:
        ocr_text = txt_path.read_text(encoding="utf-8")
        if not ocr_text.strip():
            return {
                "file_name": txt_path.name,
                "ocr_path": str(txt_path),
                "error": "empty_ocr"
            }

        extraction = extract_metadata(ocr_text)
        record = extraction.model_dump(mode="json")  # ✅ dates → ISO strings
        record["file_name"] = txt_path.name
        record["ocr_path"] = str(txt_path)
        return record

    except Exception as e:
        return {
            "file_name": txt_path.name,
            "ocr_path": str(txt_path),
            "error": f"exception: {e.__class__.__name__}"
        }

# -----------------------------------------------------------------------------
def batch_extract_features(txt_dir: Path, out_file: Path, limit: int | None = None):
    """
    Parcourt tous les .txt d'un dossier, lance extract_features_from_txt,
    et écrit les résultats dans un seul fichier JSONL.
    """
    txt_files = sorted(txt_dir.glob("*.txt"))
    total = len(txt_files)

    if limit is not None and limit < total:
        txt_files = txt_files[:limit]
        print(f"⚙️ Limiting extraction to {limit} documents (out of {total} total)")
    else:
        print(f"⚙️ Processing all {total} documents")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_file, "w", encoding="utf-8") as f_out:
        for txt_path in tqdm(txt_files, desc="🧠 Batch extraction", unit="doc"):
            try:
                record = extract_features_from_txt(txt_path)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"⚠️ Error on {txt_path.name}: {e}")

    print(f"✅ Extraction complète → {count} documents traités")
    print(f"📊 Résultats enregistrés dans: {out_file}")
