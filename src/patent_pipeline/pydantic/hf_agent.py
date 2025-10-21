import json, re, torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import ValidationError
from .models import PatentExtraction, PatentMetadata
from .prompt_templates import PROMPT_EXTRACTION
from ..utils.device_utils import get_device

HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

_model = None
_tokenizer = None
_pipe = None

def _extract_json(text: str) -> str:
    m = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
    return m.group(0) if m else "{}"

def _load_model():
    global _model, _tokenizer, _pipe
    if _pipe is not None:
        return _pipe
    device = get_device()
    print(f"🚀 Loading {HF_MODEL} on {device}")
    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    # float32 pour stabilité MPS ; sur CUDA/A100 mets torch.float16
    dtype = torch.float16 if device == "cuda" else torch.float32
    map_arg = device if device in {"cuda","mps"} else "cpu"
    _model = AutoModelForCausalLM.from_pretrained(HF_MODEL, torch_dtype=dtype, device_map=map_arg)
    _pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_new_tokens=512,
        temperature=0,
        do_sample=False,
        device=0 if device in {"cuda","mps"} else -1,
    )
    return _pipe

def extract_metadata(ocr_text: str) -> PatentExtraction:
    pipe = _load_model()
    prompt = PROMPT_EXTRACTION.format(text=ocr_text)
    out = pipe(prompt)[0]
    txt = out.get("generated_text") or out.get("text") or ""
    js = _extract_json(txt)
    try:
        data = json.loads(js)
        meta = PatentMetadata(**data)
    except (json.JSONDecodeError, ValidationError):
        meta = PatentMetadata(identifier="unknown")
    return PatentExtraction(ocr_text=ocr_text, model=HF_MODEL, prediction=meta)
