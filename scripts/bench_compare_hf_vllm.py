#!/usr/bin/env python3
"""
Compare Transformers vs vLLM on the extraction stage.

This runner delegates the actual extraction runs to bench_run_extract.py so both
backends produce the same run artifacts (preds.jsonl + run.json), then writes a
compact comparison summary with throughput and basic validity signals.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _summarize_run(run_dir: Path, *, wall_clock_s: float) -> Dict[str, Any]:
    preds_path = run_dir / "preds.jsonl"
    run_json = run_dir / "run.json"
    preds = _load_jsonl(preds_path)
    meta = _load_json(run_json)

    error_docs = sum(1 for row in preds if row.get("error"))
    valid_json_docs = sum(
        1
        for row in preds
        if not row.get("error") and isinstance(row.get("prediction"), dict)
    )
    docs = len(preds)
    docs_per_s = (docs / wall_clock_s) if wall_clock_s > 0 else 0.0

    return {
        "run_dir": str(run_dir),
        "preds_jsonl": str(preds_path),
        "run_json": str(run_json),
        "wall_clock_s": round(wall_clock_s, 6),
        "docs": docs,
        "error_docs": error_docs,
        "valid_json_docs": valid_json_docs,
        "docs_per_s": round(docs_per_s, 6),
        "mean_confidence": float(meta.get("strategy_stats", {}).get("mean_confidence", 0.0) or 0.0),
        "config": meta.get("config", {}),
        "strategy_stats": meta.get("strategy_stats", {}),
    }


def _run_bench(script_path: Path, argv: List[str]) -> float:
    cmd = [sys.executable, str(script_path), *argv]
    print("▶️", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return max(0.0, time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Transformers vs vLLM extraction runs.")

    ap.add_argument("--texts-dir", type=str, required=False)
    ap.add_argument("--ocr-run-dir", type=str, required=False)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--compare-name", type=str, required=True)

    ap.add_argument("--model-name", type=str, default=None)
    ap.add_argument("--prompt-id", type=str, default=None)
    ap.add_argument("--prompt-template-path", type=str, default=None)
    ap.add_argument("--max-ocr-chars", type=int, default=10000)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--extraction-mode", type=str, default="auto", choices=["single", "chunked", "auto"])
    ap.add_argument("--chunk-size-chars", type=int, default=7000)
    ap.add_argument("--chunk-overlap-chars", type=int, default=800)
    ap.add_argument("--extraction-passes", type=int, default=2)
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
    ap.add_argument("--save-strategy-meta", action="store_true")
    ap.add_argument("--save-raw-output", action="store_true")
    ap.add_argument("--timings", type=str, default="off", choices=["off", "basic", "detailed"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--hf-backend", type=str, default="pytorch", choices=["pytorch"])
    ap.add_argument("--hf-device", type=str, default=None)
    ap.add_argument("--hf-device-map", type=str, default=None)
    ap.add_argument("--hf-torch-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--hf-quantization", type=str, default="none", choices=["none", "bnb_8bit", "bnb_4bit"])
    ap.add_argument(
        "--hf-attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "flash_attention_2"],
    )
    ap.add_argument(
        "--hf-cache-implementation",
        type=str,
        default="auto",
        choices=["auto", "dynamic", "static", "offloaded", "offloaded_static"],
    )

    ap.add_argument("--vllm-torch-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--vllm-enable-prefix-caching", action="store_true")
    ap.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--vllm-max-model-len", type=int, default=None)
    ap.add_argument("--vllm-swap-space", type=float, default=4.0)
    ap.add_argument("--vllm-enforce-eager", action="store_true")
    ap.add_argument("--vllm-doc-batch-size", type=int, default=32)
    ap.set_defaults(vllm_sort_by_prompt_length=True)
    ap.add_argument("--vllm-sort-by-prompt-length", dest="vllm_sort_by_prompt_length", action="store_true")
    ap.add_argument("--no-vllm-sort-by-prompt-length", dest="vllm_sort_by_prompt_length", action="store_false")
    ap.add_argument("--vllm-tokenizer-mode", type=str, default="auto", choices=["auto", "mistral"])

    args = ap.parse_args()

    if not args.texts_dir and not args.ocr_run_dir:
        raise SystemExit("Provide one of --texts-dir or --ocr-run-dir")
    if args.prompt_id and args.prompt_template_path:
        raise SystemExit("Use either --prompt-id OR --prompt-template-path (not both).")

    repo_root = Path(__file__).resolve().parents[1]
    bench_script = repo_root / "scripts" / "bench_run_extract.py"
    out_root = Path(args.out_root)
    compare_dir = out_root / args.compare_name
    compare_dir.mkdir(parents=True, exist_ok=True)

    common_argv: List[str] = [
        "--out-root",
        str(out_root),
        "--max-ocr-chars",
        str(args.max_ocr_chars),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--extraction-mode",
        args.extraction_mode,
        "--chunk-size-chars",
        str(args.chunk_size_chars),
        "--chunk-overlap-chars",
        str(args.chunk_overlap_chars),
        "--extraction-passes",
        str(args.extraction_passes),
        "--strategy",
        args.strategy,
        "--header-lines",
        str(args.header_lines),
        "--targeted-rerun-threshold",
        str(args.targeted_rerun_threshold),
        "--self-consistency-n",
        str(args.self_consistency_n),
        "--self-consistency-temp",
        str(args.self_consistency_temp),
        "--merge-policy",
        args.merge_policy,
        "--timings",
        args.timings,
    ]
    if args.texts_dir:
        common_argv.extend(["--texts-dir", args.texts_dir])
    if args.ocr_run_dir:
        common_argv.extend(["--ocr-run-dir", args.ocr_run_dir])
    if args.model_name is not None:
        common_argv.extend(["--model-name", args.model_name])
    if args.prompt_id:
        common_argv.extend(["--prompt-id", args.prompt_id])
    if args.prompt_template_path:
        common_argv.extend(["--prompt-template-path", args.prompt_template_path])
    if args.do_sample:
        common_argv.append("--do-sample")
    if args.save_strategy_meta:
        common_argv.append("--save-strategy-meta")
    if args.save_raw_output:
        common_argv.append("--save-raw-output")
    if args.limit is not None:
        common_argv.extend(["--limit", str(args.limit)])
    if args.force:
        common_argv.append("--force")

    hf_run_name = f"{args.compare_name}__hf"
    hf_argv = [
        *common_argv,
        "--run-name",
        hf_run_name,
        "--backend",
        args.hf_backend,
        "--torch-dtype",
        args.hf_torch_dtype,
        "--quantization",
        args.hf_quantization,
        "--attn-implementation",
        args.hf_attn_implementation,
        "--cache-implementation",
        args.hf_cache_implementation,
    ]
    if args.hf_device is not None:
        hf_argv.extend(["--device", args.hf_device])
    if args.hf_device_map is not None:
        hf_argv.extend(["--device-map", args.hf_device_map])
    vllm_run_name = f"{args.compare_name}__vllm"
    vllm_argv = [
        *common_argv,
        "--run-name",
        vllm_run_name,
        "--backend",
        "vllm",
        "--torch-dtype",
        args.vllm_torch_dtype,
        "--quantization",
        "none",
        "--attn-implementation",
        "auto",
        "--cache-implementation",
        "auto",
        "--vllm-tensor-parallel-size",
        str(args.vllm_tensor_parallel_size),
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm-swap-space",
        str(args.vllm_swap_space),
        "--vllm-doc-batch-size",
        str(args.vllm_doc_batch_size),
        "--vllm-tokenizer-mode",
        args.vllm_tokenizer_mode,
    ]
    if args.vllm_enable_prefix_caching:
        vllm_argv.append("--vllm-enable-prefix-caching")
    if not args.vllm_sort_by_prompt_length:
        vllm_argv.append("--no-vllm-sort-by-prompt-length")
    if args.vllm_max_model_len is not None:
        vllm_argv.extend(["--vllm-max-model-len", str(args.vllm_max_model_len)])
    if args.vllm_enforce_eager:
        vllm_argv.append("--vllm-enforce-eager")

    hf_elapsed = _run_bench(bench_script, hf_argv)
    vllm_elapsed = _run_bench(bench_script, vllm_argv)

    hf_summary = _summarize_run(out_root / hf_run_name, wall_clock_s=hf_elapsed)
    vllm_summary = _summarize_run(out_root / vllm_run_name, wall_clock_s=vllm_elapsed)

    hf_docs_per_s = float(hf_summary["docs_per_s"] or 0.0)
    vllm_docs_per_s = float(vllm_summary["docs_per_s"] or 0.0)
    speedup = (vllm_docs_per_s / hf_docs_per_s) if hf_docs_per_s > 0 else None

    summary = {
        "kind": "slm_compare",
        "created_at_unix": time.time(),
        "compare_name": args.compare_name,
        "out_root": str(out_root),
        "hf_run_name": hf_run_name,
        "vllm_run_name": vllm_run_name,
        "runs": {
            "transformers": hf_summary,
            "vllm": vllm_summary,
        },
        "comparison": {
            "same_docs": hf_summary["docs"] == vllm_summary["docs"],
            "same_valid_json_docs": hf_summary["valid_json_docs"] == vllm_summary["valid_json_docs"],
            "same_error_docs": hf_summary["error_docs"] == vllm_summary["error_docs"],
            "vllm_speedup_docs_per_s": round(speedup, 6) if speedup is not None else None,
            "valid_json_delta": int(vllm_summary["valid_json_docs"]) - int(hf_summary["valid_json_docs"]),
            "error_docs_delta": int(vllm_summary["error_docs"]) - int(hf_summary["error_docs"]),
            "mean_confidence_delta": round(
                float(vllm_summary["mean_confidence"]) - float(hf_summary["mean_confidence"]),
                6,
            ),
        },
    }

    compare_json = compare_dir / "compare.json"
    compare_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Compare done")
    print("Compare dir:", compare_dir)
    print("Summary:", compare_json)
    print("HF run:", out_root / hf_run_name)
    print("vLLM run:", out_root / vllm_run_name)


if __name__ == "__main__":
    main()
