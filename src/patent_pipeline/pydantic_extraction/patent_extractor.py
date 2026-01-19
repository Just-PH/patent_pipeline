# 📄 src/patent_pipeline/pydantic_extraction/patent_extractor.py
"""
PatentExtractor

Rôle:
- Charger un modèle (MLX ou PyTorch/Transformers)
- Construire un prompt (via prompt_id v1/v2/v3 OU un template fourni)
- Générer une sortie
- Extraire un JSON du texte généré
- Normaliser + valider via Pydantic (PatentMetadata)
- (Option) mesurer des timings par document (off/basic/detailed)
- Écrire des records JSONL consommables par la suite (scoring)

Note importante (benchmark):
- prompt_hash doit être STABLE par run → on hash le TEMPLATE + suffix fixe,
  PAS le prompt final (qui inclut l'OCR et changerait par document).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import regex as re
import torch
from pydantic import ValidationError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .models import PatentExtraction, PatentMetadata
from .prompt_templates import PROMPT_EXTRACTION_V1, PROMPT_EXTRACTION_V2, PROMPT_EXTRACTION_V3
from ..utils.device_utils import get_device

# Optional deps (MLX on Apple Silicon)
try:
    import mlx_lm

    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False


# ---------------------------------------------------------------------------
# Prompt registry (matrice 3D : prompts = [v1, v2, v3])
# ---------------------------------------------------------------------------
_PROMPT_BY_ID: Dict[str, str] = {
    "v1": PROMPT_EXTRACTION_V1,
    "v2": PROMPT_EXTRACTION_V2,
    "v3": PROMPT_EXTRACTION_V3,
}

_JSON_ONLY_SUFFIX = "\n\nNow output ONLY the JSON object, without any extra text.\n"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class PatentExtractor:
    """
    Extracteur de métadonnées de brevets à partir de textes OCR.

    Supporte:
    - MLX (Apple Silicon) : rapide en local dev sur Mac
    - PyTorch (CPU/CUDA)  : pour VM/H100

    La dimension "prompt" devient une vraie dimension de benchmark via:
    - prompt_id: "v1"/"v2"/"v3"
      OU
    - prompt_template: template string contenant "{text}"
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: Literal["auto", "mlx", "pytorch"] = "auto",
        device: Optional[str] = None,
        # Prompt selection
        prompt_id: Optional[str] = None,
        prompt_template: Optional[str] = None,
        # Generation params
        max_ocr_chars: int = 10000,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        # Timings
        timings: Literal["off", "basic", "detailed"] = "off",
    ):
        # ----------------------------
        # Basic config
        # ----------------------------
        self.model_name = model_name or os.getenv("HF_MODEL", "mlx-community/Mistral-7B-Instruct-v0.3")
        self.max_ocr_chars = max_ocr_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.timings = timings

        # Constant suffix (part of prompt hash)
        self.prompt_suffix = _JSON_ONLY_SUFFIX

        # ----------------------------
        # Prompt config (matrice 3D)
        # ----------------------------
        self.prompt_id = prompt_id

        if self.prompt_id is not None:
            if self.prompt_id not in _PROMPT_BY_ID:
                raise ValueError(f"Unknown prompt_id={self.prompt_id!r}. Allowed: {sorted(_PROMPT_BY_ID.keys())}")
            self.prompt_template = _PROMPT_BY_ID[self.prompt_id]
            self.prompt_template_source = f"prompt_id:{self.prompt_id}"
        else:
            self.prompt_template = prompt_template or PROMPT_EXTRACTION_V2
            self.prompt_template_source = "inline_template" if prompt_template else "default:v2"

        if "{text}" not in self.prompt_template:
            raise ValueError("Le template doit contenir le placeholder {text}")

        # ✅ STABLE par run: template + suffix (pas le prompt rendu avec OCR)
        self.prompt_hash = _sha256(self.prompt_template + self.prompt_suffix)

        # ----------------------------
        # Backend selection
        # ----------------------------
        if backend == "auto":
            self.backend = "mlx" if _HAS_MLX else "pytorch"
        else:
            self.backend = backend
            if self.backend == "mlx" and not _HAS_MLX:
                raise ImportError("MLX n'est pas installé. Installe avec: pip install mlx-lm")

        # ----------------------------
        # Device selection (PyTorch)
        # ----------------------------
        self.device = device or get_device()

        # NOTE: MPS est souvent instable pour Transformers “text-generation”.
        if self.backend == "pytorch" and self.device == "mps":
            print("⚠️  MPS backend instable → fallback CPU")
            self.device = "cpu"

        # last per-document timings set by extract()
        self._last_timing: Optional[Dict[str, float]] = None

        # Model objects
        self.model = None
        self.tokenizer = None
        self.pipe = None

        # ----------------------------
        # Load model
        # ----------------------------
        self._load_model()

    # =========================================================================
    # Model loading / generation
    # =========================================================================

    def _load_model(self) -> None:
        """Charge le modèle selon le backend configuré."""
        print(f"🧠 Backend: {self.backend}")
        print(f"📦 Model: {self.model_name}")

        if self.backend == "mlx":
            print("⚙️  Loading via MLX")
            self.model, self.tokenizer = mlx_lm.load(self.model_name)
            self.pipe = None
            return

        if self.backend == "pytorch":
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            map_arg = self.device if self.device == "cuda" else "cpu"

            print(f"🚀 Loading on {self.device} ({dtype})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=map_arg,
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            return

        raise ValueError(f"Unknown backend: {self.backend}")

    def _generate(self, prompt: str) -> str:
        """Génère du texte avec le modèle chargé."""
        if self.backend == "mlx":
            # mlx_lm.generate signature can vary; keep it simple/reliable.
            output = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=self.max_new_tokens,
            )
            return (output or "").strip()

        if self.backend == "pytorch":
            out = self.pipe(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
            )[0]
            return out.get("generated_text") or out.get("text") or ""

        raise ValueError(f"Unknown backend: {self.backend}")

    # =========================================================================
    # Prompt helpers
    # =========================================================================

    def set_prompt_template(self, template: str) -> None:
        """
        Permet de changer le template de prompt “à la main”.
        IMPORTANT: recalcule prompt_hash (sinon bug silencieux).
        """
        if "{text}" not in template:
            raise ValueError("Le template doit contenir le placeholder {text}")
        self.prompt_id = None
        self.prompt_template = template
        self.prompt_template_source = "inline_template"
        self.prompt_hash = _sha256(self.prompt_template + self.prompt_suffix)

    def _truncate_ocr(self, text: str) -> str:
        """Tronque le texte OCR si trop long (évite prompts gigantesques)."""
        if len(text) > self.max_ocr_chars:
            return text[: self.max_ocr_chars] + "\n[...] (truncated)"
        return text

    # =========================================================================
    # Parsing / normalization
    # =========================================================================

    def _extract_json(self, text: str) -> str:
        """
        Extrait le premier bloc JSON valide du texte généré.

        Stratégie:
        1) trouver un bloc {...} équilibré (regex récursive)
        2) fallback sur une extraction partielle à partir de "identifier"
        3) sinon {} (échec)
        """
        m = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
        if m:
            return m.group(0)

        alt = re.search(r'"identifier".*', text, re.DOTALL)
        if alt:
            raw = alt.group(0).strip()
            if not raw.startswith("{"):
                raw = "{\n" + raw
            if not raw.endswith("}"):
                raw += "\n}"
            return raw

        return "{}"

    def _normalize_entity_list(self, value):
        """Normalise inventors/assignees."""
        if value is None:
            return None

        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return value

        if isinstance(value, str):
            entities = []
            for chunk in value.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                m = re.match(r"(.+?)\s*\(([^)]+)\)", chunk)
                if m:
                    entities.append({"name": m.group(1).strip(), "address": m.group(2).strip()})
                else:
                    entities.append({"name": chunk, "address": None})
            return entities if entities else None

        return None

    def _is_company_name(self, name: str) -> bool:
        """Heuristique: détecte si un nom ressemble à une entreprise."""
        if not name:
            return False

        name_lower = name.lower()
        company_patterns = [
            r"\&",
            r"\bund\b",
            r"\bet\b",
            r"\bgmbh\b",
            r"\bag\b",
            r"\bsa\b",
            r"\bco\.",
            r"\bkg\b",
            r"\bltd\b",
            r"\binc\b",
            r"\bcorp\b",
            r"\bs\.a\.",
            r"\bs\.r\.l\.",
        ]
        for pat in company_patterns:
            if re.search(pat, name_lower):
                return True

        letters = [c for c in name if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.6:
                return True

        return False

    def _fix_inventor_assignee_confusion(self, data: dict) -> dict:
        """Si des entreprises apparaissent dans inventors, les déplacer vers assignees."""
        inventors = data.get("inventors") or []
        assignees = data.get("assignees") or []

        if not inventors:
            return data

        true_inventors = []
        misplaced_companies = []

        for inv in inventors:
            if isinstance(inv, dict):
                name = inv.get("name", "")
                if self._is_company_name(name):
                    misplaced_companies.append(inv)
                else:
                    true_inventors.append(inv)

        if misplaced_companies:
            print(f"🔧 Correction : {len(misplaced_companies)} entreprise(s) déplacée(s) vers assignees")
            data["inventors"] = true_inventors if true_inventors else None
            data["assignees"] = (assignees + misplaced_companies) or None

        return data

    def _fix_duplicate_dates(self, data: dict) -> dict:
        """Corrige des dates dupliquées (heuristique pragmatique)."""
        app_date = data.get("pub_date_application")
        pub_date = data.get("pub_date_publication")
        foreign_date = data.get("pub_date_foreign")

        if app_date and pub_date and app_date == pub_date:
            data["pub_date_application"] = None

        if app_date and pub_date and foreign_date and app_date == pub_date == foreign_date:
            data["pub_date_application"] = None
            data["pub_date_foreign"] = None

        return data

    def _parse_and_validate(self, json_str: str) -> PatentMetadata:
        """Parse le JSON et valide avec Pydantic."""
        try:
            data = json.loads(json_str)

            if not isinstance(data, dict):
                print(f"⚠️ JSON type inattendu: {type(data)}")
                data = data[0] if isinstance(data, list) and data else {}

            # rétrocompatibilité (noms de champs)
            if "assignee" in data and "assignees" not in data:
                data["assignees"] = data.pop("assignee")
            if "inventor" in data and "inventors" not in data:
                data["inventors"] = data.pop("inventor")
            if "class" in data and "classification" not in data:
                data["classification"] = data.pop("class")

            # champs requis (même si null)
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
            for k in required_fields:
                data.setdefault(k, None)

            # normalisation inventors/assignees
            data["inventors"] = self._normalize_entity_list(data.get("inventors"))
            data["assignees"] = self._normalize_entity_list(data.get("assignees"))

            # corrections heuristiques
            data = self._fix_inventor_assignee_confusion(data)
            data = self._fix_duplicate_dates(data)

            return PatentMetadata(**data)

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            print(f"⚠️  Erreur de validation JSON: {e}")
            print(f"→ JSON brut:\n{json_str}\n")
            return PatentMetadata(identifier="unknown")

    # =========================================================================
    # Extraction (with timings)
    # =========================================================================

    def extract(self, ocr_text: str, debug: bool = False) -> PatentExtraction:
        """
        Extrait les métadonnées structurées d'un texte OCR.

        Timings:
        - basic: t_generate_s, t_total_s
        - detailed: + t_prompt_s, t_parse_s
        """
        t0 = time.perf_counter() if self.timings != "off" else None

        # 1) Troncature OCR
        truncated_text = self._truncate_ocr(ocr_text)

        # 2) Construction du prompt
        t_prompt0 = time.perf_counter() if self.timings == "detailed" else None
        prompt = self.prompt_template.format(text=truncated_text) + self.prompt_suffix
        t_prompt1 = time.perf_counter() if self.timings == "detailed" else None

        if debug:
            print("=" * 80)
            print("📝 PROMPT ENVOYÉ AU MODÈLE:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        # 3) Génération
        t_gen0 = time.perf_counter() if self.timings != "off" else None
        raw_output = self._generate(prompt)
        t_gen1 = time.perf_counter() if self.timings != "off" else None

        if debug:
            print("\n" + "=" * 80)
            print("🤖 SORTIE BRUTE DU MODÈLE:")
            print("=" * 80)
            print(raw_output)
            print("=" * 80)

        # 4) Parsing JSON + validation Pydantic
        t_parse0 = time.perf_counter() if self.timings == "detailed" else None
        json_str = self._extract_json(raw_output)

        if debug:
            print("\n" + "=" * 80)
            print("📦 JSON EXTRAIT:")
            print("=" * 80)
            print(json_str)
            print("=" * 80 + "\n")

        metadata = self._parse_and_validate(json_str)
        t_parse1 = time.perf_counter() if self.timings == "detailed" else None

        # 5) Timings dict (persistable)
        t_end = time.perf_counter() if self.timings != "off" else None
        self._last_timing = self._timing_dict(
            t0=t0,
            t_prompt0=t_prompt0,
            t_prompt1=t_prompt1,
            t_gen0=t_gen0,
            t_gen1=t_gen1,
            t_parse0=t_parse0,
            t_parse1=t_parse1,
            t_end=t_end,
        )

        return PatentExtraction(
            ocr_text=ocr_text,
            model=self.model_name,
            prediction=metadata,
        )

    def _timing_dict(
        self,
        *,
        t0: Optional[float],
        t_prompt0: Optional[float],
        t_prompt1: Optional[float],
        t_gen0: Optional[float],
        t_gen1: Optional[float],
        t_parse0: Optional[float],
        t_parse1: Optional[float],
        t_end: Optional[float],
    ) -> Optional[Dict[str, float]]:
        if self.timings == "off" or t0 is None or t_end is None:
            return None

        out: Dict[str, float] = {}
        out["t_total_s"] = max(0.0, t_end - t0)

        if t_gen0 is not None and t_gen1 is not None:
            out["t_generate_s"] = max(0.0, t_gen1 - t_gen0)

        if self.timings == "detailed":
            if t_prompt0 is not None and t_prompt1 is not None:
                out["t_prompt_s"] = max(0.0, t_prompt1 - t_prompt0)
            if t_parse0 is not None and t_parse1 is not None:
                out["t_parse_s"] = max(0.0, t_parse1 - t_parse0)

        return out

    # =========================================================================
    # File-level wrapper (JSONL record)
    # =========================================================================

    def extract_from_file(self, txt_path: Path) -> dict:
        """
        Extrait les métadonnées d'un fichier .txt.

        Returns:
            Dict sérialisable en JSON (prêt pour JSONL).
            Ajoute prompt_id/prompt_hash et timing si dispo.
        """
        t_file0 = time.perf_counter() if self.timings != "off" else None

        try:
            # Lecture OCR
            t_read0 = time.perf_counter() if self.timings == "detailed" else None
            ocr_text = txt_path.read_text(encoding="utf-8", errors="ignore")
            t_read1 = time.perf_counter() if self.timings == "detailed" else None

            if not ocr_text.strip():
                rec = {"file_name": txt_path.name, "ocr_path": str(txt_path), "error": "empty_ocr"}
                if self.prompt_id is not None:
                    rec["prompt_id"] = self.prompt_id
                rec["prompt_hash"] = self.prompt_hash

                if self.timings != "off" and t_file0 is not None:
                    timing = {"t_total_file_s": time.perf_counter() - t_file0}
                    if self.timings == "detailed" and t_read0 is not None and t_read1 is not None:
                        timing["t_read_s"] = max(0.0, t_read1 - t_read0)
                    rec["timing"] = timing

                return rec

            # Extraction LLM
            extraction = self.extract(ocr_text)
            record = extraction.model_dump(mode="json")

            # Champs utiles pour le bench
            record["file_name"] = txt_path.name
            record["ocr_path"] = str(txt_path)

            # Identifier depuis le nom de fichier
            if isinstance(record.get("prediction"), dict):
                record["prediction"]["identifier"] = txt_path.stem.split("_")[0]

            # Dimensions de benchmark
            if self.prompt_id is not None:
                record["prompt_id"] = self.prompt_id
            record["prompt_hash"] = self.prompt_hash

            # Timings
            if self.timings != "off":
                timing_out: Dict[str, float] = {}
                if self.timings == "detailed" and t_read0 is not None and t_read1 is not None:
                    timing_out["t_read_s"] = max(0.0, t_read1 - t_read0)

                if self._last_timing:
                    timing_out.update(self._last_timing)

                if t_file0 is not None:
                    timing_out["t_total_file_s"] = max(0.0, time.perf_counter() - t_file0)

                record["timing"] = timing_out

            return record

        except Exception as e:
            print(f"⚠️ Erreur sur {txt_path.name}: {e}")
            traceback.print_exc()

            rec = {"file_name": txt_path.name, "ocr_path": str(txt_path), "error": f"exception: {e.__class__.__name__}"}
            if self.prompt_id is not None:
                rec["prompt_id"] = self.prompt_id
            rec["prompt_hash"] = self.prompt_hash

            if self.timings != "off" and t_file0 is not None:
                rec["timing"] = {"t_total_file_s": max(0.0, time.perf_counter() - t_file0)}

            return rec

    # =========================================================================
    # Batch runner
    # =========================================================================

    def batch_extract(self, txt_dir: Path, out_file: Path, limit: Optional[int] = None) -> int:
        """
        Traite un dossier de fichiers .txt en batch.

        Returns:
            nombre de documents traités
        """
        txt_files = sorted(txt_dir.glob("*.txt"))
        total = len(txt_files)

        if limit is not None and limit < total:
            txt_files = txt_files[:limit]
            print(f"⚙️ Limitation à {limit} documents (sur {total} total)")
        else:
            print(f"⚙️ Traitement de {total} documents")

        out_file.parent.mkdir(parents=True, exist_ok=True)
        count = 0

        with open(out_file, "w", encoding="utf-8") as f_out:
            for txt_path in tqdm(txt_files, desc="🧠 Batch extraction", unit="doc"):
                record = self.extract_from_file(txt_path)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"✅ Extraction complète → {count} documents traités")
        print(f"📊 Résultats: {out_file}")

        return count
