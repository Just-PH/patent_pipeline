#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from patent_pipeline.pydantic_extraction.patent_extractor import PatentExtractor


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _as_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Path -> str recursively (avoid PosixPath not JSON serializable)."""
    def conv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, list):
            return [conv(v) for v in x]
        return x
    return conv(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts-dir", type=str, required=False, help="Directory with *.txt (identifier.txt).")
    ap.add_argument("--ocr-run-dir", type=str, required=False, help="output/ocr/<run>/ (expects texts/ inside).")

    ap.add_argument("--out-root", type=str, required=True, help="Root output dir, e.g. output/slm")
    ap.add_argument("--run-name", type=str, required=True, help="Name of this extraction run")

    ap.add_argument("--model-name", type=str, default=None)
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "mlx", "pytorch"])
    ap.add_argument("--prompt-template-path", type=str, default=None)

    ap.add_argument("--max-ocr-chars", type=int, default=10000)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    if not args.texts_dir and not args.ocr_run_dir:
        raise SystemExit("Provide one of --texts-dir or --ocr-run-dir")

    if args.ocr_run_dir:
        ocr_run_dir = Path(args.ocr_run_dir)
        texts_dir = ocr_run_dir / "texts"
    else:
        texts_dir = Path(args.texts_dir)

    if not texts_dir.exists():
        raise FileNotFoundError(f"Missing texts dir: {texts_dir}")

    out_root = Path(args.out_root)
    run_dir = out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_jsonl = run_dir / "preds.jsonl"
    run_json = run_dir / "run.json"

    if pred_jsonl.exists() and not args.force:
        raise SystemExit(f"Pred file exists: {pred_jsonl} (use --force to overwrite)")

    prompt_template: Optional[str] = None
    if args.prompt_template_path:
        prompt_template = _read_text(Path(args.prompt_template_path))
        if "{text}" not in prompt_template:
            raise ValueError("Prompt template must contain {text}")

    # --- extractor ---
    extractor = PatentExtractor(
        model_name=args.model_name,
        backend=args.backend,
        prompt_template=prompt_template,
        max_ocr_chars=args.max_ocr_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        device=args.device,
    )

    meta = {
        "kind": "slm_extract",
        "created_at_unix": time.time(),
        "texts_dir": texts_dir,
        "out_root": out_root,
        "run_name": args.run_name,
        "pred_jsonl": pred_jsonl,
        "config": {
            "model_name": extractor.model_name,
            "backend": extractor.backend,
            "max_ocr_chars": args.max_ocr_chars,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": bool(args.do_sample),
            "device": extractor.device,
            "limit": args.limit,
            "prompt_template_path": args.prompt_template_path,
        },
    }
    run_json.write_text(json.dumps(_as_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    # --- run ---
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
