#!/usr/bin/env python3
"""
Benchmark runner — Extraction stage (SLM)

But:
- Lire des fichiers OCR .txt (depuis --texts-dir ou --ocr-run-dir/texts)
- Lancer l'extracteur SLM (PatentExtractor)
- Écrire un JSONL (preds.jsonl) + un run.json

Support matrice 3D:
- OCR × SLM × PROMPT via --prompt-id (v1/v2/v3/v4)
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
from typing import Any, Dict, List, Optional

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


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * q))
    idx = min(max(idx, 0), len(arr) - 1)
    return float(arr[idx])


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
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "mlx", "pytorch", "vllm"])
    ap.add_argument("--device", type=str, default=None, help="cpu/cuda/mps (only relevant for pytorch)")
    ap.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Transformers device_map for pytorch backend (e.g. cuda|auto).",
    )
    ap.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="PyTorch model dtype policy. auto keeps backend defaults; others force dtype.",
    )
    ap.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "bnb_8bit", "bnb_4bit"],
        help="Optional model quantization for pytorch backend.",
    )
    ap.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "flash_attention_2"],
        help="Attention backend for pytorch backend.",
    )
    ap.add_argument(
        "--cache-implementation",
        type=str,
        default="auto",
        choices=["auto", "dynamic", "static", "offloaded", "offloaded_static"],
        help="KV cache strategy for generate() on pytorch backend.",
    )
    ap.add_argument(
        "--vllm-enable-prefix-caching",
        action="store_true",
        help="Enable vLLM automatic prefix caching.",
    )
    ap.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM backend.",
    )
    ap.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization target for vLLM backend.",
    )
    ap.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional max_model_len override for vLLM backend.",
    )
    ap.add_argument(
        "--vllm-swap-space",
        type=float,
        default=4.0,
        help="Swap space in GiB for vLLM backend.",
    )
    ap.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Force eager execution in vLLM.",
    )
    ap.add_argument(
        "--vllm-doc-batch-size",
        type=int,
        default=32,
        help="Application-level micro-batch size (number of documents) for vLLM baseline batching.",
    )
    ap.set_defaults(vllm_sort_by_prompt_length=True)
    ap.add_argument(
        "--vllm-sort-by-prompt-length",
        dest="vllm_sort_by_prompt_length",
        action="store_true",
        help="Sort prompts by length before vLLM micro-batching.",
    )
    ap.add_argument(
        "--no-vllm-sort-by-prompt-length",
        dest="vllm_sort_by_prompt_length",
        action="store_false",
        help="Keep original document order when forming vLLM micro-batches.",
    )
    ap.add_argument(
        "--vllm-tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "mistral"],
        help="Tokenizer mode for vLLM backend.",
    )

    # ----------------------------
    # Prompt selection (matrice 3D)
    # ----------------------------
    ap.add_argument(
        "--prompt-id",
        type=str,
        default=None,
        help="Prompt ID (e.g. v1, v2, v3, v4) resolved via prompt_templates.py",
    )
    ap.add_argument(
        "--prompt-template-path",
        type=str,
        default=None,
        help="Path to a prompt template file containing {text} (mutually exclusive with --prompt-id)",
    )
    ap.add_argument(
        "--guardrail-profile",
        type=str,
        default="auto",
        choices=["auto", "off", "de_legacy_self_applicant"],
        help="Optional post-generation guardrail profile. auto keeps current prompt-driven defaults.",
    )

    # ----------------------------
    # Generation params
    # ----------------------------
    ap.add_argument("--max-ocr-chars", type=int, default=10000)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument(
        "--extraction-mode",
        type=str,
        default="auto",
        choices=["single", "chunked", "auto"],
        help="single=legacy truncate+one pass, chunked=always chunk, auto=chunk only when text > max-ocr-chars.",
    )
    ap.add_argument("--chunk-size-chars", type=int, default=7000)
    ap.add_argument("--chunk-overlap-chars", type=int, default=800)
    ap.add_argument("--extraction-passes", type=int, default=2, help="Number of chunking passes with shifted boundaries.")
    ap.add_argument(
        "--strategy",
        type=str,
        default="baseline",
        choices=["baseline", "chunked", "header_first", "two_pass_targeted", "self_consistency"],
    )
    ap.add_argument("--header-lines", type=int, default=30)
    ap.add_argument("--targeted-rerun-threshold", type=float, default=0.6)
    ap.add_argument("--self-consistency-n", type=int, default=2)
    ap.add_argument("--self-consistency-temp", type=float, default=0.2)
    ap.add_argument(
        "--merge-policy",
        type=str,
        default="prefer_non_null",
        choices=["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"],
    )
    ap.add_argument("--save-strategy-meta", action="store_true", help="Persist strategy metadata in preds.jsonl.")

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
    ap.add_argument(
        "--save-raw-output",
        action="store_true",
        help="Persist full model raw output per document under run_dir/raw_outputs/ and store path in preds.jsonl.",
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
    raw_output_dir = run_dir / "raw_outputs" if args.save_raw_output else None

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
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        quantization=args.quantization,
        attn_implementation=args.attn_implementation,
        cache_implementation=args.cache_implementation,
        vllm_enable_prefix_caching=bool(args.vllm_enable_prefix_caching),
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_swap_space=args.vllm_swap_space,
        vllm_enforce_eager=bool(args.vllm_enforce_eager),
        vllm_doc_batch_size=args.vllm_doc_batch_size,
        vllm_sort_by_prompt_length=bool(args.vllm_sort_by_prompt_length),
        vllm_tokenizer_mode=args.vllm_tokenizer_mode,
        # prompt selection
        prompt_id=args.prompt_id,
        prompt_template=prompt_template,
        guardrail_profile=args.guardrail_profile,
        # gen params
        max_ocr_chars=args.max_ocr_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=bool(args.do_sample),
        extraction_mode=args.extraction_mode,
        chunk_size_chars=args.chunk_size_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        extraction_passes=args.extraction_passes,
        strategy=args.strategy,
        header_lines=args.header_lines,
        targeted_rerun_threshold=args.targeted_rerun_threshold,
        self_consistency_n=args.self_consistency_n,
        self_consistency_temp=args.self_consistency_temp,
        merge_policy=args.merge_policy,
        save_strategy_meta=bool(args.save_strategy_meta),
        save_raw_output=bool(args.save_raw_output),
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
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
            "quantization": args.quantization,
            "attn_implementation": args.attn_implementation,
            "cache_implementation": args.cache_implementation,
            "vllm_enable_prefix_caching": bool(args.vllm_enable_prefix_caching),
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_max_model_len": args.vllm_max_model_len,
            "vllm_swap_space": args.vllm_swap_space,
            "vllm_enforce_eager": bool(args.vllm_enforce_eager),
            "vllm_doc_batch_size": args.vllm_doc_batch_size,
            "vllm_sort_by_prompt_length": bool(args.vllm_sort_by_prompt_length),
            "vllm_tokenizer_mode": args.vllm_tokenizer_mode,
            # Prompt identity (run-stable)
            "prompt_id": extractor.prompt_id,
            "guardrail_profile": extractor.guardrail_profile,
            "prompt_template_path": args.prompt_template_path,
            "prompt_template_source": getattr(extractor, "prompt_template_source", None),
            "prompt_hash": getattr(extractor, "prompt_hash", None),
            # Generation params
            "max_ocr_chars": args.max_ocr_chars,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": bool(args.do_sample),
            "extraction_mode": args.extraction_mode,
            "chunk_size_chars": args.chunk_size_chars,
            "chunk_overlap_chars": args.chunk_overlap_chars,
            "extraction_passes": args.extraction_passes,
            "strategy": args.strategy,
            "header_lines": args.header_lines,
            "targeted_rerun_threshold": args.targeted_rerun_threshold,
            "self_consistency_n": args.self_consistency_n,
            "self_consistency_temp": args.self_consistency_temp,
            "merge_policy": args.merge_policy,
            "save_strategy_meta": bool(args.save_strategy_meta),
            # Timings
            "timings": args.timings,
            "save_raw_output": bool(args.save_raw_output),
            "raw_output_dir": str(raw_output_dir) if raw_output_dir is not None else None,
            "limit": args.limit,
        },
        "strategy_version": "v1",
        "strategy_config": {
            "strategy": args.strategy,
            "extraction_mode": args.extraction_mode,
            "chunk_size_chars": args.chunk_size_chars,
            "chunk_overlap_chars": args.chunk_overlap_chars,
            "extraction_passes": args.extraction_passes,
            "header_lines": args.header_lines,
            "targeted_rerun_threshold": args.targeted_rerun_threshold,
            "self_consistency_n": args.self_consistency_n,
            "self_consistency_temp": args.self_consistency_temp,
            "merge_policy": args.merge_policy,
            "save_strategy_meta": bool(args.save_strategy_meta),
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
        raw_output_dir=raw_output_dir,
    )

    preds = _load_jsonl(pred_jsonl)
    confidence_values = [float(x.get("confidence_score", 0.0) or 0.0) for x in preds]
    rerun_values = [bool(x.get("was_rerun", False)) for x in preds]
    rerun_rate = (sum(1 for x in rerun_values if x) / len(rerun_values)) if rerun_values else 0.0
    strategy_stats = {
        "docs": len(preds),
        "docs_rerun_pct": round(rerun_rate * 100.0, 4),
        "mean_confidence": round(sum(confidence_values) / len(confidence_values), 6) if confidence_values else 0.0,
        "confidence_distribution": {
            "p10": round(_percentile(confidence_values, 0.10), 6),
            "p50": round(_percentile(confidence_values, 0.50), 6),
            "p90": round(_percentile(confidence_values, 0.90), 6),
        },
    }
    meta["strategy_stats"] = strategy_stats
    run_json.write_text(json.dumps(_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Extraction done")
    print("Run dir:", run_dir)
    print("Preds:", pred_jsonl)
    if raw_output_dir is not None:
        print("Raw outputs:", raw_output_dir)
    print("Run meta:", run_json)
    print("Docs:", n)


if __name__ == "__main__":
    main()
