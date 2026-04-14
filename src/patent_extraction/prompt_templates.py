from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

PROMPT_EXTRACTION_V4 = (PROMPTS_DIR / "de_legacy_v4.txt").read_text(encoding="utf-8")

PROMPT_BY_ID = {
    "v4": PROMPT_EXTRACTION_V4,
}

JSON_ONLY_SUFFIX = "\n\nNow output ONLY the JSON object, without any extra text.\n"


__all__ = ["JSON_ONLY_SUFFIX", "PROMPT_BY_ID", "PROMPT_EXTRACTION_V4", "PROMPTS_DIR"]
