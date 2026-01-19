#!/usr/bin/env python3
"""
Benchmark runner — Extraction stage (SLM)

But:
- Lire des fichiers OCR .txt (depuis --texts-dir ou --ocr-run-dir/texts)
- Lancer l'extracteur SLM (PatentExtractor)
- Écrire un JSONL (preds.jsonl) + un run.json

Support matrice 3D:
- OCR × SLM × PROMPT via --prompt-id (v1/v2/v3)
- Compatibilité: --prompt-template-path (mutuellement exclusif avec --prompt-id)

Note benchmark/repro:
- On persist prompt_id + prompt_hash (hash du TEMPLATE + suffix fixe)
  → stable au niveau run (pas doc), utile pour cache/traçabilité.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from patent_pipeline.pydantic_extraction.patent_extractor import PatentExtractor


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _jsonable(x: Any) -> Any:
    """Convert Path -> str recursively for JSON serialization."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark runner — Extraction stage (writes preds.jsonl + run.json)",
    )

    # ----------------------------
    # Inputs: soit un texts-dir, soit un ocr-run-dir
    # ----------------------------
    ap.add_argument("--texts-dir", type=str, required=False, help="Directory with *.txt (identifier.txt).")
    ap.add_argument("--ocr-run-dir", type=str, required=False, help="output/ocr/<run>/ (expects texts/ inside).")

    # ----------------------------
    # Outputs
    # ----------------------------
    ap.add_argument("--out-root", type=str, required=True, help="Root output dir, e.g. output/slm")
    ap.add_argument("--run-name", type=str, required=True, help="Name of this extraction run")

    # ----------------------------
    # Model / backend
    # ----------------------------
    ap.add_argument("--model-name", type=str, default=None, help="HF repo id or local path; defaults to env HF_MODEL.")
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "mlx", "pytorch"])
    ap.add_argument("--device", type=str, default=None, help="cpu/cuda/mps (only relevant for pytorch)")

    # ----------------------------
    # Prompt selection (matrice 3D)
    # ----------------------------
    ap.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Prompt ID (e.g. v1, v2, v3) resolved via prompt_templates.py",
    )
    ap.add_argument(
        "--prompt-template-path",
        type=str,
        default=None,
        help="Path to a prompt template file containing {text} (mutually exclusive with --prompt-id)",
    )

    # ----------------------------
    # Generation params
    # ----------------------------
    ap.add_argument("--max-ocr-chars", type=int, default=10000)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--do-sample", action="store_true")

    # ----------------------------
    # Timings + misc
    # ----------------------------
    ap.add_argument(
        "--timings",
        type=str,
        default="off",
        choices=["off", "basic", "detailed"],
        help="Timing instrumentation level (off/basic/detailed).",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    # ----------------------------
    # Validate inputs
    # ----------------------------
    if not args.texts_dir and not args.ocr_run_dir:
        raise SystemExit("Provide one of --texts-dir or --ocr-run-dir")

    if args.prompt_id and args.prompt_template_path:
        raise SystemExit("Use either --prompt-id OR --prompt-template-path (not both).")

    # Resolve texts_dir
    if args.ocr_run_dir:
        texts_dir = Path(args.ocr_run_dir) / "texts"
    else:
        texts_dir = Path(args.texts_dir)

    if not texts_dir.exists():
        raise FileNotFoundError(f"Missing texts dir: {texts_dir}")

    # Resolve output run dir
    out_root = Path(args.out_root)
    run_dir = out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = run_dir / "preds.jsonl"
    run_json = run_dir / "run.json"

    if pred_jsonl.exists() and not args.force:
        raise SystemExit(f"Pred file exists: {pred_jsonl} (use --force to overwrite)")

    # If a prompt file is provided, read it once and pass the template string to the extractor.
    prompt_template: Optional[str] = None
    if args.prompt_template_path:
        prompt_template = _read_text(Path(args.prompt_template_path))
        if "{text}" not in prompt_template:
            raise ValueError("Prompt template must contain {text}")

    # ----------------------------
    # Create extractor
    # ----------------------------
    extractor = PatentExtractor(
        model_name=args.model_name,
        backend=args.backend,
        device=args.device,
        # prompt selection
        prompt_id=args.prompt_id,
        prompt_template=prompt_template,
        # gen params
        max_ocr_chars=args.max_ocr_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=bool(args.do_sample),
        # timings
        timings=args.timings,
    )

    # ----------------------------
    # Persist run metadata (for reproducibility)
    # ----------------------------
    meta: Dict[str, Any] = {
        "kind": "slm_extract",
        "created_at_unix": time.time(),
        "texts_dir": str(texts_dir),
        "out_root": str(out_root),
        "run_name": args.run_name,
        "preds_jsonl": str(pred_jsonl),
        "config": {
            # SLM identity
            "model_name": extractor.model_name,
            "backend": extractor.backend,
            "device": extractor.device,
            # Prompt identity (run-stable)
            "prompt_id": extractor.prompt_id,
            "prompt_template_path": args.prompt_template_path,
            "prompt_template_source": getattr(extractor, "prompt_template_source", None),
            "prompt_hash": getattr(extractor, "prompt_hash", None),
            # Generation params
            "max_ocr_chars": args.max_ocr_chars,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": bool(args.do_sample),
            # Timings
            "timings": args.timings,
            "limit": args.limit,
        },
    }

    run_json.write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    # ----------------------------
    # Run extraction
    # ----------------------------
    n = extractor.batch_extract(
        txt_dir=texts_dir,
        out_file=pred_jsonl,
        limit=args.limit,
    )

    print("✅ Extraction done")
    print("Run dir:", run_dir)
    print("Preds:", pred_jsonl)
    print("Run meta:", run_json)
    print("Docs:", n)


if __name__ == "__main__":
    main()
