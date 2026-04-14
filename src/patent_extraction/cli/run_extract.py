from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from patent_extraction.extractor import PatentExtractionRunner
from patent_extraction.profiles import DEFAULT_PROFILE_NAME


def _add_bool_override_flags(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    help_enable: str,
    help_disable: str,
) -> None:
    parser.set_defaults(**{name: None})
    parser.add_argument(f"--{name.replace('_', '-')}", dest=name, action="store_true", help=help_enable)
    parser.add_argument(f"--no-{name.replace('_', '-')}", dest=name, action="store_false", help=help_disable)


def _collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    keys = [
        "model_name",
        "torch_dtype",
        "quantization",
        "prompt_id",
        "guardrail_profile",
        "max_ocr_chars",
        "max_new_tokens",
        "temperature",
        "do_sample",
        "save_strategy_meta",
        "save_raw_output",
        "timings",
        "enable_prefix_caching",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "swap_space",
        "enforce_eager",
        "doc_batch_size",
        "sort_by_prompt_length",
        "tokenizer_mode",
        "strategy",
        "extraction_mode",
        "chunk_size_chars",
        "chunk_overlap_chars",
        "extraction_passes",
        "header_lines",
        "targeted_rerun_threshold",
        "self_consistency_n",
        "self_consistency_temp",
        "merge_policy",
    ]
    return {key: getattr(args, key) for key in keys if hasattr(args, key) and getattr(args, key) is not None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run extraction using the clean patent_extraction package.")
    parser.add_argument("--texts-dir", type=Path, required=True, help="Directory containing OCR text files.")
    parser.add_argument("--out-root", type=Path, required=True, help="Root directory for run outputs.")
    parser.add_argument("--run-name", type=str, required=True, help="Run directory name inside out-root.")
    parser.add_argument("--profile", type=str, default=DEFAULT_PROFILE_NAME, help="Profile name under profile_defs/.")
    parser.add_argument("--profile-path", type=Path, default=None, help="Optional explicit profile definition path.")
    parser.add_argument("--prompt-path", type=Path, default=None, help="Optional prompt file overriding the profile.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of docs to process.")
    parser.add_argument("--force", action="store_true", help="Overwrite preds/run metadata if run dir exists.")

    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None, choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--quantization", type=str, default=None, help="vLLM quantization method, e.g. bitsandbytes, awq, gptq.")
    parser.add_argument("--prompt-id", type=str, default=None, help="Packaged prompt id. Currently v4 is supported.")
    parser.add_argument(
        "--guardrail-profile",
        type=str,
        default=None,
        choices=["auto", "off", "de_legacy_self_applicant"],
    )
    parser.add_argument("--max-ocr-chars", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timings", type=str, default=None, choices=["off", "basic", "detailed"])

    _add_bool_override_flags(
        parser,
        name="do_sample",
        help_enable="Enable sampling.",
        help_disable="Disable sampling.",
    )
    _add_bool_override_flags(
        parser,
        name="save_strategy_meta",
        help_enable="Persist strategy metadata in predictions.",
        help_disable="Do not persist strategy metadata.",
    )
    _add_bool_override_flags(
        parser,
        name="save_raw_output",
        help_enable="Persist raw model outputs.",
        help_disable="Do not persist raw model outputs.",
    )
    _add_bool_override_flags(
        parser,
        name="enable_prefix_caching",
        help_enable="Enable vLLM prefix caching.",
        help_disable="Disable vLLM prefix caching.",
    )
    _add_bool_override_flags(
        parser,
        name="enforce_eager",
        help_enable="Enable vLLM eager mode.",
        help_disable="Disable vLLM eager mode.",
    )
    _add_bool_override_flags(
        parser,
        name="sort_by_prompt_length",
        help_enable="Sort prompts by length before vLLM batching.",
        help_disable="Keep original order before vLLM batching.",
    )

    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--swap-space", type=float, default=None)
    parser.add_argument("--doc-batch-size", type=int, default=None)
    parser.add_argument("--tokenizer-mode", type=str, default=None, choices=["auto", "mistral"])

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["baseline", "chunked", "header_first", "two_pass_targeted", "self_consistency"],
    )
    parser.add_argument("--extraction-mode", type=str, default=None, choices=["single", "chunked", "auto"])
    parser.add_argument("--chunk-size-chars", type=int, default=None)
    parser.add_argument("--chunk-overlap-chars", type=int, default=None)
    parser.add_argument("--extraction-passes", type=int, default=None)
    parser.add_argument("--header-lines", type=int, default=None)
    parser.add_argument("--targeted-rerun-threshold", type=float, default=None)
    parser.add_argument("--self-consistency-n", type=int, default=None)
    parser.add_argument("--self-consistency-temp", type=float, default=None)
    parser.add_argument(
        "--merge-policy",
        type=str,
        default=None,
        choices=["prefer_non_null", "prefer_first", "prefer_last", "vote_majority"],
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runner = PatentExtractionRunner.from_profile(
        name=args.profile,
        profile_path=args.profile_path,
        prompt_path=args.prompt_path,
        overrides=_collect_overrides(args),
    )
    artifacts = runner.batch_extract(
        texts_dir=args.texts_dir,
        out_root=args.out_root,
        run_name=args.run_name,
        limit=args.limit,
        force=args.force,
    )

    summary = {
        "run_dir": str(artifacts.run_dir),
        "preds_path": str(artifacts.preds_path),
        "run_meta_path": str(artifacts.run_meta_path),
        "raw_outputs_dir": str(artifacts.raw_outputs_dir) if artifacts.raw_outputs_dir is not None else None,
        "docs_written": artifacts.docs_written,
        "wall_s": round(artifacts.wall_s, 3),
        "profile": runner.profile.name,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
