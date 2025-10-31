# 📄 src/patent_pipeline/pydantic/hf_agent.py
from pathlib import Path
import traceback
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
HF_MODEL = os.getenv("HF_MODEL", "mlx-community/Mistral-7B-Instruct-v0.3")

_model = None
_tokenizer = None
_pipe = None
_USE_MLX = False

# -----------------------------------------------------------------------------
def _extract_json(text: str) -> str:
    """
    Extract the first valid JSON-like block from the model output.
    Falls back to a minimal JSON object if not found.
    """
    # Try to find the most complete {...} block
    m = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
    if m:
        return m.group(0)

    # Fallback: try to find partial JSON (starting with quotes)
    alt = re.search(r'"identifier".*', text, re.DOTALL)
    if alt:
        raw = alt.group(0).strip()
        # ensure braces
        if not raw.startswith("{"):
            raw = "{\n" + raw
        if not raw.endswith("}"):
            raw += "\n}"
        return raw

    # Default to empty JSON
    return "{}"
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
def _normalize_entity_list(value):
    """
    Normalize a list or string of entities (inventors/assignees).
    Example inputs:
        "Jean Dupont (Paris); Marie Curie (Versailles)"
        [{"name": "Jean Dupont", "address": "Paris"}]
    Returns a list of dicts with fields: name, address
    """
    if value is None:
        return None

    # Already good
    if isinstance(value, list) and all(isinstance(x, dict) for x in value):
        return value

    # If it's a string → split by semicolon
    if isinstance(value, str):
        entities = []
        for chunk in value.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            # Try to extract "Name (City)"
            m = re.match(r"(.+?)\s*\(([^)]+)\)", chunk)
            if m:
                entities.append({"name": m.group(1).strip(), "address": m.group(2).strip()})
            else:
                entities.append({"name": chunk, "address": None})
        return entities

    return None
# -----------------------------------------------------------------------------
def extract_metadata(ocr_text: str) -> PatentExtraction:
    """Main entrypoint: run the prompt, parse and validate with Pydantic."""
    pipe = _load_model()
    # ⚙️ Truncate OCR to avoid cutting off instructions
    MAX_CHARS = 4000
    if len(ocr_text) > MAX_CHARS:
        ocr_text = ocr_text[:MAX_CHARS] + "\n[...] (truncated)"
    prompt = PROMPT_EXTRACTION.format(text=ocr_text)
    prompt += "\n\nNow output ONLY the JSON object, without any extra text.\n"
    # 1️⃣ Run inference
    if _USE_MLX:
        output = mlx_lm.generate(_model, _tokenizer, prompt, max_tokens=1024)
        txt = output.strip()
    else:
        out = pipe(prompt)[0]
        txt = out.get("generated_text") or out.get("text") or ""

    # 2️⃣ Extract JSON and clean
    js = _extract_json(txt)

    try:
        data = json.loads(js)
        if not isinstance(data, dict):
            print(f"⚠️ Unexpected JSON type: {type(data)}")
            data = data[0] if isinstance(data, list) and data else {}


        # 🧩 Backward compatibility mapping
        if "assignee" in data and "assignees" not in data:
            data["assignees"] = data.pop("assignee")
        if "inventor" in data and "inventors" not in data:
            data["inventors"] = data.pop("inventor")
        if "class" in data and "classification" not in data:
            data["classification"] = data.pop("class")

        # ✅ Ensure all required keys exist (avoid KeyError)
        required_fields = [
            "title",
            "inventors",
            "assignees",
            "pub_date_application",
            "pub_date_publication",
            "pub_date_foreign",
            "classification",
            "industrial_field",
        ]
        for key in required_fields:
            data.setdefault(key, None)

        # ✅ Normalize lists
        data["inventors"] = _normalize_entity_list(data.get("inventors"))
        data["assignees"] = _normalize_entity_list(data.get("assignees"))

        meta = PatentMetadata(**data)

    except (json.JSONDecodeError, ValidationError, KeyError) as e:
        print(f"⚠️  JSON validation error: {e}")
        print(f"→ Raw JSON returned by model:\n{js}\n")
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
        record["prediction"]["identifier"] = txt_path.stem.split("_")[0]
        return record

    except Exception as e:
        print(f"⚠️ Error processing {txt_path.name}: {e}")
        traceback.print_exc()
        return {
            "file_name": txt_path.name,
            "ocr_path": str(txt_path),
            "error": f"exception: {e.__class__.__name__}"
        }

# -----------------------------------------------------------------------------
def batch_extract_features(txt_dir: Path, out_file: Path, limit_llm: int | None = None):
    """
    Parcourt tous les .txt d'un dossier, lance extract_features_from_txt,
    et écrit les résultats dans un seul fichier JSONL.
    """
    txt_files = sorted(txt_dir.glob("*.txt"))
    total = len(txt_files)

    if limit_llm is not None and limit_llm < total:
        txt_files = txt_files[:limit_llm]
        print(f"⚙️ Limiting extraction to {limit_llm} documents (out of {total} total)")
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
