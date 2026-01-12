from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Literal, Iterable
import json
import os
import traceback
import inspect

import regex as re
from pydantic import ValidationError

from patent_pipeline.pydantic_extraction.models import PatentExtraction, PatentMetadata
from patent_pipeline.pydantic_extraction.prompt_templates import PROMPT_EXTRACTION


# =============================================================================
# Utils
# =============================================================================

_JSON_RECURSIVE = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)


def extract_first_json_block(text: str) -> str:
    m = _JSON_RECURSIVE.search(text or "")
    if m:
        return m.group(0)

    alt = re.search(r'"identifier"\s*:.*', text or "", re.DOTALL)
    if alt:
        raw = alt.group(0).strip()
        if not raw.startswith("{"):
            raw = "{\n" + raw
        if not raw.endswith("}"):
            raw += "\n}"
        return raw

    return "{}"


def ocr_text_cleanup_basic(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(\p{L}{2,})-\s*\n\s*(\p{L}{2,})", r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_entities(value: Any) -> Optional[List[Dict[str, Optional[str]]]]:
    if value is None:
        return None

    if isinstance(value, list) and all(isinstance(x, dict) for x in value):
        return value

    if isinstance(value, str):
        out: List[Dict[str, Optional[str]]] = []
        for chunk in value.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            m = re.match(r"(.+?)\s*\(([^)]+)\)", chunk)
            if m:
                out.append({"name": m.group(1).strip(), "address": m.group(2).strip()})
            else:
                out.append({"name": chunk, "address": None})
        return out or None

    return None


def chunked(seq: Sequence[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield list(seq[i:i+n])


# =============================================================================
# Small config objects (no dataclasses)
# =============================================================================

InputPolicy = Literal["first_chars", "first_lines"]
BackendKind = Literal["auto", "mlx", "transformers"]


class InputConfig:
    def __init__(
        self,
        *,
        cleanup_ocr: bool = True,
        max_chars: int = 4000,
        max_lines: int = 160,
        policy: InputPolicy = "first_lines",
    ):
        self.cleanup_ocr = cleanup_ocr
        self.max_chars = max_chars
        self.max_lines = max_lines
        self.policy = policy

    def apply(self, text: str) -> str:
        t = ocr_text_cleanup_basic(text) if self.cleanup_ocr else (text or "")
        if self.policy == "first_chars":
            return t[: self.max_chars] if self.max_chars and len(t) > self.max_chars else t
        # first_lines
        lines = t.splitlines()
        if self.max_lines and len(lines) > self.max_lines:
            lines = lines[: self.max_lines]
        return "\n".join(lines).strip()


class GenerationConfig:
    def __init__(
        self,
        *,
        max_new_tokens: int = 600,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p


# =============================================================================
# Backend interface
# =============================================================================

class LLMBackend(Protocol):
    name: str
    is_gpu: bool
    supports_batch: bool

    def generate(self, prompt: str, *, gen: GenerationConfig) -> str: ...
    def generate_batch(self, prompts: List[str], *, gen: GenerationConfig) -> List[str]: ...


# =============================================================================
# Backends
# =============================================================================

class TransformersBackend:
    supports_batch = True

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,       # "cpu" | "cuda" | "mps" | None
        torch_dtype: Optional[str] = None,  # "float16" | "float32" | None
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype

        self._tokenizer = None
        self._model = None

        self.name = model_id
        self.is_gpu = False

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = (self.device or os.getenv("HF_DEVICE") or "cpu").lower()
        if device not in {"cpu", "cuda", "mps"}:
            device = "cpu"

        dtype = torch.float32
        if self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "float32":
            dtype = torch.float32
        else:
            if device == "cuda":
                dtype = torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype)

        if device != "cpu":
            self._model = self._model.to(device)
            self.is_gpu = True

    def generate(self, prompt: str, *, gen: GenerationConfig) -> str:
        return self.generate_batch([prompt], gen=gen)[0]

    def generate_batch(self, prompts: List[str], *, gen: GenerationConfig) -> List[str]:
        self._load()
        import torch

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        do_sample = gen.temperature > 0.0

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                temperature=gen.temperature if do_sample else None,
                top_p=gen.top_p if do_sample else None,
                do_sample=do_sample,
            )

        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # strip prompt (important)
        cleaned: List[str] = []
        for p, full in zip(prompts, decoded):
            cleaned.append(full[len(p):].lstrip() if full.startswith(p) else full)
        return cleaned


class MLXBackend:
    supports_batch = False

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.name = model_id
        self.is_gpu = True
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import mlx_lm
        self._model, self._tokenizer = mlx_lm.load(self.model_id)

    def generate(self, prompt: str, *, gen: GenerationConfig) -> str:
        self._load()
        import mlx_lm

        # API mlx_lm.generate varie selon versions: temp vs temperature, etc.
        sig = inspect.signature(mlx_lm.generate)
        params = sig.parameters

        kwargs: Dict[str, Any] = {}

        # max tokens
        if "max_tokens" in params:
            kwargs["max_tokens"] = gen.max_new_tokens

        # top_p
        if "top_p" in params:
            kwargs["top_p"] = gen.top_p

        # temperature: n'envoie rien si 0 pour forcer greedy sur certaines versions
        if gen.temperature and gen.temperature > 0:
            if "temperature" in params:
                kwargs["temperature"] = gen.temperature
            elif "temp" in params:
                kwargs["temp"] = gen.temperature

        # prompt arg name
        if "prompt" in params:
            kwargs["prompt"] = prompt
            out = mlx_lm.generate(self._model, self._tokenizer, **kwargs)
        else:
            # fallback (rare)
            out = mlx_lm.generate(self._model, self._tokenizer, prompt, **kwargs)

        return out or ""

    def generate_batch(self, prompts: List[str], *, gen: GenerationConfig) -> List[str]:
        raise NotImplementedError("MLXBackend: pas de vrai batch ici (pour l'instant).")


# =============================================================================
# Result object (debug friendly)
# =============================================================================

class ExtractionResult:
    def __init__(
        self,
        *,
        ok: bool,
        prediction: Optional[PatentMetadata],
        raw_output: str,
        error: Optional[str] = None,
        error_details: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        self.ok = ok
        self.prediction = prediction
        self.raw_output = raw_output
        self.error = error
        self.error_details = error_details
        self.prompt = prompt


# =============================================================================
# ONE class: model + prompt + parsing + pydantic
# =============================================================================

class PatentExtractor:
    """
    ONE class encapsulating:
      - backend selection (mlx/transformers/auto)
      - prompt template
      - generation config
      - JSON parsing + Pydantic validation

    Usage:
      extractor = PatentExtractor(model_id="mlx-community/...", backend="mlx")
      res = extractor.extract_one(ocr_text)  # ExtractionResult
      ex  = extractor.extract(ocr_text)      # PatentExtraction
    """

    def __init__(
        self,
        *,
        model_id: str,
        backend: BackendKind = "auto",
        device: Optional[str] = None,        # for transformers
        torch_dtype: Optional[str] = None,   # for transformers
        prompt_template: str = PROMPT_EXTRACTION,
        input_cfg: Optional[InputConfig] = None,
        gen_cfg: Optional[GenerationConfig] = None,
        batch_size: int = 8,
        verbose: bool = False,
    ):
        self.model_id = model_id
        self.backend_kind = backend
        self.device = device
        self.torch_dtype = torch_dtype

        self.prompt_template = prompt_template
        self.input_cfg = input_cfg or InputConfig()
        self.gen_cfg = gen_cfg or GenerationConfig()
        self.batch_size = batch_size
        self.verbose = verbose

        self._backend: Optional[LLMBackend] = None

    # --------------------
    # Backend
    # --------------------
    @property
    def backend(self) -> LLMBackend:
        if self._backend is None:
            self._backend = self._make_backend()
        return self._backend

    def _make_backend(self) -> LLMBackend:
        prefer = (self.backend_kind or "auto").lower().strip()

        if prefer in {"auto", "mlx"}:
            try:
                import mlx_lm  # noqa: F401
                return MLXBackend(self.model_id)
            except Exception as e:
                if prefer == "mlx":
                    raise RuntimeError(
                        "Backend MLX demandé mais mlx_lm indisponible ou modèle non chargeable.\n"
                        "Installe: pip install mlx-lm\n"
                        f"Erreur originale: {e}"
                    )

        # fallback transformers
        return TransformersBackend(self.model_id, device=self.device, torch_dtype=self.torch_dtype)

    # --------------------
    # Prompt
    # --------------------
    def build_prompt(self, ocr_text: str) -> str:
        txt = self.input_cfg.apply(ocr_text)
        tpl = self.prompt_template
        return tpl.format(ocr_text=txt) if "{ocr_text}" in tpl else f"{tpl}\n\nOCR TEXT:\n{txt}\n"

    # --------------------
    # Parsing
    # --------------------
    def parse_prediction(self, raw: str) -> PatentMetadata:
        js = extract_first_json_block(raw)
        data = json.loads(js) if js else {}

        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            data = {}

        # backward compat keys
        for old, new in [("assignee", "assignees"), ("inventor", "inventors"), ("class", "classification")]:
            if old in data and new not in data:
                data[new] = data.pop(old)

        # ensure keys exist
        for k in [
            "title", "inventors", "assignees",
            "pub_date_application", "pub_date_publication",
            "pub_date_foreign", "classification", "industrial_field"
        ]:
            data.setdefault(k, None)

        data["inventors"] = normalize_entities(data.get("inventors"))
        data["assignees"] = normalize_entities(data.get("assignees"))

        return PatentMetadata(**data)

    # --------------------
    # Single
    # --------------------
    def extract_one(self, ocr_text: str) -> ExtractionResult:
        prompt = self.build_prompt(ocr_text)
        raw = ""

        try:
            if self.verbose:
                print("----- PROMPT(head) -----")
                print(prompt[:800])

            raw = self.backend.generate(prompt, gen=self.gen_cfg)

            if self.verbose:
                print("----- RAW(head) -----")
                print((raw or "")[:800])

            if not (raw or "").strip():
                return ExtractionResult(
                    ok=False, prediction=None, raw_output=raw,
                    error="empty_output", error_details="LLM returned empty string",
                    prompt=prompt,
                )

            pred = self.parse_prediction(raw)
            return ExtractionResult(ok=True, prediction=pred, raw_output=raw, prompt=prompt)

        except ValidationError as e:
            return ExtractionResult(
                ok=False, prediction=None, raw_output=raw,
                error="pydantic_validation", error_details=str(e),
                prompt=prompt,
            )
        except Exception as e:
            return ExtractionResult(
                ok=False, prediction=None, raw_output=raw,
                error="exception", error_details=f"{e}\n{traceback.format_exc()}",
                prompt=prompt,
            )

    def extract(self, ocr_text: str) -> PatentExtraction:
        r = self.extract_one(ocr_text)
        pred = r.prediction if r.prediction is not None else self._empty_metadata()
        return PatentExtraction(ocr_text=ocr_text, model=self.backend.name, prediction=pred)

    # --------------------
    # Batch
    # --------------------
    def extract_batch_one(self, ocr_texts: List[str]) -> List[ExtractionResult]:
        prompts = [self.build_prompt(t) for t in ocr_texts]
        gen = self.gen_cfg

        outputs: List[str] = []
        if getattr(self.backend, "supports_batch", False):
            for chunk in chunked(prompts, max(1, self.batch_size)):
                outputs.extend(self.backend.generate_batch(chunk, gen=gen))
        else:
            outputs = [self.backend.generate(p, gen=gen) for p in prompts]

        results: List[ExtractionResult] = []
        for prompt, raw in zip(prompts, outputs):
            try:
                if not (raw or "").strip():
                    results.append(ExtractionResult(
                        ok=False, prediction=None, raw_output=raw,
                        error="empty_output", error_details="LLM returned empty string",
                        prompt=prompt,
                    ))
                    continue
                pred = self.parse_prediction(raw)
                results.append(ExtractionResult(ok=True, prediction=pred, raw_output=raw, prompt=prompt))
            except ValidationError as e:
                results.append(ExtractionResult(ok=False, prediction=None, raw_output=raw, error="pydantic_validation", error_details=str(e), prompt=prompt))
            except Exception as e:
                results.append(ExtractionResult(ok=False, prediction=None, raw_output=raw, error="exception", error_details=f"{e}\n{traceback.format_exc()}", prompt=prompt))

        return results

    def extract_batch(self, ocr_texts: List[str]) -> List[PatentExtraction]:
        rs = self.extract_batch_one(ocr_texts)
        out: List[PatentExtraction] = []
        for ocr_text, r in zip(ocr_texts, rs):
            pred = r.prediction if r.prediction is not None else self._empty_metadata()
            out.append(PatentExtraction(ocr_text=ocr_text, model=self.backend.name, prediction=pred))
        return out

    # --------------------
    # Helpers
    # --------------------
    @staticmethod
    def _empty_metadata() -> PatentMetadata:
        return PatentMetadata(
            title=None,
            inventors=None,
            assignees=None,
            pub_date_application=None,
            pub_date_publication=None,
            pub_date_foreign=None,
            classification=None,
            industrial_field=None,
        )
