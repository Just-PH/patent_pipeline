# 📄 src/patent_pipeline/pydantic/patent_extractor.py
from pathlib import Path
import traceback
import os
import json
import regex as re
from typing import Optional, Literal
from pydantic import ValidationError
from .models import PatentExtraction, PatentMetadata
from .prompt_templates import PROMPT_EXTRACTION_V2
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


class PatentExtractor:
    """
    Extracteur de métadonnées de brevets à partir de textes OCR.

    Supporte MLX (Apple Silicon) et PyTorch (CPU/CUDA).
    Utilise un LLM pour extraire des champs structurés via Pydantic.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: Literal["auto", "mlx", "pytorch"] = "auto",
        prompt_template: Optional[str] = None,
        max_ocr_chars: int = 10000,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialise l'extracteur avec un modèle LLM.

        Args:
            model_name: Nom du modèle HuggingFace (défaut: env HF_MODEL ou Mistral-7B)
            backend: "mlx" (Mac), "pytorch" (CPU/CUDA), ou "auto" (détection)
            prompt_template: Template de prompt personnalisé (défaut: PROMPT_EXTRACTION_V2)
            max_ocr_chars: Nombre max de caractères OCR à envoyer au modèle
            max_new_tokens: Tokens max générés par le modèle
            temperature: Température de génération (0 = déterministe)
            do_sample: Active le sampling (False pour reproductibilité)
            device: Device PyTorch ('cpu', 'cuda', 'mps'), auto-détecté si None
        """
        self.model_name = model_name or os.getenv("HF_MODEL", "mlx-community/Mistral-7B-Instruct-v0.3")
        self.prompt_template = prompt_template or PROMPT_EXTRACTION_V2
        self.max_ocr_chars = max_ocr_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        # Détection du backend
        if backend == "auto":
            self.backend = "mlx" if _HAS_MLX else "pytorch"
        else:
            self.backend = backend
            if backend == "mlx" and not _HAS_MLX:
                raise ImportError("MLX n'est pas installé. Installe avec: pip install mlx-lm")

        # Détection du device pour PyTorch
        self.device = device or get_device()
        if self.backend == "pytorch" and self.device == "mps":
            print("⚠️  MPS backend instable → fallback CPU")
            self.device = "cpu"

        # Chargement du modèle
        self._load_model()

    def _load_model(self):
        """Charge le modèle selon le backend configuré."""
        print(f"🧠 Backend: {self.backend}")
        print(f"📦 Model: {self.model_name}")

        if self.backend == "mlx":
            print("⚙️  Loading via MLX (quantized int4/int8)")
            self.model, self.tokenizer = mlx_lm.load(self.model_name)
            self.pipe = None

        elif self.backend == "pytorch":
            # Config PyTorch pour stabilité MPS
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            map_arg = self.device if self.device == "cuda" else "cpu"

            print(f"🚀 Loading on {self.device} ({dtype})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=map_arg
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample
            )

    def set_prompt_template(self, template: str):
        """
        Change le template de prompt utilisé pour l'extraction.

        Args:
            template: Nouveau template avec placeholder {text}
        """
        if "{text}" not in template:
            raise ValueError("Le template doit contenir le placeholder {text}")
        self.prompt_template = template

    def _truncate_ocr(self, text: str) -> str:
        """Tronque le texte OCR si trop long."""
        if len(text) > self.max_ocr_chars:
            return text[:self.max_ocr_chars] + "\n[...] (truncated)"
        return text

    def _generate(self, prompt: str) -> str:
        """
        Génère du texte avec le modèle chargé.

        Args:
            prompt: Prompt complet à envoyer au modèle

        Returns:
            Texte généré brut
        """
        if self.backend == "mlx":
            output = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=self.max_new_tokens
            )
            return output.strip()

        elif self.backend == "pytorch":
            out = self.pipe(prompt)[0]
            return out.get("generated_text") or out.get("text") or ""

    def _extract_json(self, text: str) -> str:
        """
        Extrait le premier bloc JSON valide du texte généré.

        Args:
            text: Texte brut du modèle

        Returns:
            Chaîne JSON extraite (ou "{}" si échec)
        """
        # Recherche du bloc {...} complet
        m = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
        if m:
            return m.group(0)

        # Fallback: recherche partielle à partir de "identifier"
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
        """
        Normalise une liste d'entités (inventeurs/assignees).

        Accepte:
        - Liste de dicts [{"name": ..., "address": ...}]
        - String "Jean Dupont (Paris); Marie Curie (Versailles)"

        Returns:
            Liste de dicts normalisés ou None
        """
        if value is None:
            return None

        # Déjà au bon format
        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return value

        # Parsing depuis string
        if isinstance(value, str):
            entities = []
            for chunk in value.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                # Extraction "Nom (Ville)"
                m = re.match(r"(.+?)\s*\(([^)]+)\)", chunk)
                if m:
                    entities.append({
                        "name": m.group(1).strip(),
                        "address": m.group(2).strip()
                    })
                else:
                    entities.append({"name": chunk, "address": None})
            return entities if entities else None

        return None

    def _is_company_name(self, name: str) -> bool:
        """
        Détecte si un nom est probablement une entreprise.

        Indices :
        - Contient &, und, et
        - Contient GmbH, AG, SA, Co., KG, Ltd, Inc
        - Tout en majuscules (>= 50% de lettres majuscules)

        Args:
            name: Nom à analyser

        Returns:
            True si c'est probablement une entreprise
        """
        if not name:
            return False

        name_lower = name.lower()

        # Patterns évidents de compagnies
        company_patterns = [
            r'\&',  # &
            r'\bund\b',  # und
            r'\bet\b',  # et
            r'\bgmbh\b',
            r'\bag\b',
            r'\bsa\b',
            r'\bco\.',
            r'\bkg\b',
            r'\bltd\b',
            r'\binc\b',
            r'\bcorp\b',
            r'\bs\.a\.',
            r'\bs\.r\.l\.',
        ]

        for pattern in company_patterns:
            if re.search(pattern, name_lower):
                return True

        # Heuristique : beaucoup de majuscules = entreprise
        letters = [c for c in name if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.6:  # Plus de 60% de majuscules
                return True

        return False

    def _fix_inventor_assignee_confusion(self, data: dict) -> dict:
        """
        Corrige automatiquement les confusions inventors/assignees.

        Si des noms d'entreprise sont dans inventors, les déplace vers assignees.

        Args:
            data: Dict avec champs inventors et assignees

        Returns:
            Dict corrigé
        """
        inventors = data.get("inventors") or []
        assignees = data.get("assignees") or []

        if not inventors:
            return data

        # Séparer vrais inventors et companies mal placées
        true_inventors = []
        misplaced_companies = []

        for inventor in inventors:
            if isinstance(inventor, dict):
                name = inventor.get("name", "")
                if self._is_company_name(name):
                    misplaced_companies.append(inventor)
                else:
                    true_inventors.append(inventor)

        # Si on a trouvé des compagnies mal placées
        if misplaced_companies:
            print(f"🔧 Correction : {len(misplaced_companies)} entreprise(s) déplacée(s) vers assignees")
            for company in misplaced_companies:
                print(f"   → {company.get('name')}")

            # Fusionner avec les assignees existants
            all_assignees = assignees + misplaced_companies

            data["inventors"] = true_inventors if true_inventors else None
            data["assignees"] = all_assignees if all_assignees else None

        return data

    def _fix_duplicate_dates(self, data: dict) -> dict:
        """
        Corrige les dates dupliquées.

        Si pub_date_application == pub_date_publication et qu'il n'y a pas de foreign date,
        on assume que c'est la date de publication, pas d'application.

        Logique :
        - Si les 3 dates sont identiques → garder seulement publication
        - Si application == publication (mais ≠ foreign) → mettre application à None
        - Si une seule date existe → c'est probablement la publication

        Args:
            data: Dict avec champs de dates

        Returns:
            Dict corrigé
        """
        app_date = data.get("pub_date_application")
        pub_date = data.get("pub_date_publication")
        foreign_date = data.get("pub_date_foreign")

        # Cas 1 : Application == Publication (duplication probable)
        if app_date and pub_date and app_date == pub_date:
            # Si foreign est différent, on garde les 3
            if foreign_date and foreign_date != app_date:
                print(f"🔧 Correction dates : application et publication identiques ({app_date})")
                print(f"   → Interprétation : {app_date} = publication (car foreign={foreign_date} existe)")
                data["pub_date_application"] = None
            else:
                # Pas de foreign ou foreign identique aussi → c'est juste la publication
                print(f"🔧 Correction dates : une seule date trouvée ({app_date})")
                print(f"   → Interprétation : {app_date} = date de publication")
                data["pub_date_application"] = None

        # Cas 2 : Les 3 dates identiques (très improbable)
        if app_date and pub_date and foreign_date and app_date == pub_date == foreign_date:
            print(f"🔧 Correction dates : 3 dates identiques ({app_date})")
            print(f"   → Garde seulement publication")
            data["pub_date_application"] = None
            data["pub_date_foreign"] = None

        return data

    def _parse_and_validate(self, json_str: str) -> PatentMetadata:
        """
        Parse le JSON et valide avec Pydantic.

        Args:
            json_str: Chaîne JSON brute

        Returns:
            PatentMetadata validé
        """
        try:
            data = json.loads(json_str)

            # Gestion des types inattendus
            if not isinstance(data, dict):
                print(f"⚠️ JSON type inattendu: {type(data)}")
                data = data[0] if isinstance(data, list) and data else {}

            # Rétrocompatibilité des noms de champs
            if "assignee" in data and "assignees" not in data:
                data["assignees"] = data.pop("assignee")
            if "inventor" in data and "inventors" not in data:
                data["inventors"] = data.pop("inventor")
            if "class" in data and "classification" not in data:
                data["classification"] = data.pop("class")

            # Initialisation des champs requis
            required_fields = [
                "title", "inventors", "assignees",
                "pub_date_application", "pub_date_publication", "pub_date_foreign",
                "classification", "industrial_field"
            ]
            for key in required_fields:
                data.setdefault(key, None)

            # Normalisation des listes
            data["inventors"] = self._normalize_entity_list(data.get("inventors"))
            data["assignees"] = self._normalize_entity_list(data.get("assignees"))

            # 🔧 CORRECTION AUTOMATIQUE : déplacer les entreprises mal classées
            data = self._fix_inventor_assignee_confusion(data)

            # 🔧 CORRECTION AUTOMATIQUE : gérer les dates dupliquées
            data = self._fix_duplicate_dates(data)

            return PatentMetadata(**data)

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            print(f"⚠️  Erreur de validation JSON: {e}")
            print(f"→ JSON brut:\n{json_str}\n")
            return PatentMetadata(identifier="unknown")

    def extract(self, ocr_text: str, debug: bool = False) -> PatentExtraction:
        """
        Extrait les métadonnées structurées d'un texte OCR.

        Args:
            ocr_text: Texte brut issu de l'OCR
            debug: Si True, affiche le prompt complet et la sortie brute

        Returns:
            PatentExtraction avec metadata Pydantic validé
        """
        # Troncature si nécessaire
        truncated_text = self._truncate_ocr(ocr_text)

        # Construction du prompt
        prompt = self.prompt_template.format(text=truncated_text)
        prompt += "\n\nNow output ONLY the JSON object, without any extra text.\n"

        if debug:
            print("=" * 80)
            print("📝 PROMPT ENVOYÉ AU MODÈLE:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        # Génération
        raw_output = self._generate(prompt)

        if debug:
            print("\n" + "=" * 80)
            print("🤖 SORTIE BRUTE DU MODÈLE:")
            print("=" * 80)
            print(raw_output)
            print("=" * 80)

        # Extraction et parsing JSON
        json_str = self._extract_json(raw_output)

        if debug:
            print("\n" + "=" * 80)
            print("📦 JSON EXTRAIT:")
            print("=" * 80)
            print(json_str)
            print("=" * 80 + "\n")

        metadata = self._parse_and_validate(json_str)

        return PatentExtraction(
            ocr_text=ocr_text,
            model=self.model_name,
            prediction=metadata
        )

    def extract_from_file(self, txt_path: Path) -> dict:
        """
        Extrait les métadonnées d'un fichier .txt.

        Args:
            txt_path: Chemin vers le fichier OCR .txt

        Returns:
            Dict sérialisable en JSON (prêt pour JSONL)
        """
        try:
            ocr_text = txt_path.read_text(encoding="utf-8")

            if not ocr_text.strip():
                return {
                    "file_name": txt_path.name,
                    "ocr_path": str(txt_path),
                    "error": "empty_ocr"
                }

            extraction = self.extract(ocr_text)
            record = extraction.model_dump(mode="json")
            record["file_name"] = txt_path.name
            record["ocr_path"] = str(txt_path)

            # Extraction de l'identifier depuis le nom du fichier
            record["prediction"]["identifier"] = txt_path.stem.split("_")[0]

            return record

        except Exception as e:
            print(f"⚠️ Erreur sur {txt_path.name}: {e}")
            traceback.print_exc()
            return {
                "file_name": txt_path.name,
                "ocr_path": str(txt_path),
                "error": f"exception: {e.__class__.__name__}"
            }

    def batch_extract(
        self,
        txt_dir: Path,
        out_file: Path,
        limit: Optional[int] = None
    ) -> int:
        """
        Traite un dossier de fichiers .txt en batch.

        Args:
            txt_dir: Dossier contenant les fichiers .txt
            out_file: Fichier JSONL de sortie
            limit: Nombre max de fichiers à traiter (None = tous)

        Returns:
            Nombre de documents traités avec succès
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
                try:
                    record = self.extract_from_file(txt_path)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    print(f"⚠️ Erreur sur {txt_path.name}: {e}")

        print(f"✅ Extraction complète → {count} documents traités")
        print(f"📊 Résultats: {out_file}")

        return count
