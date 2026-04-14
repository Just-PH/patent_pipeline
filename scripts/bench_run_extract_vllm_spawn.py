#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    script_path = "/app/scripts/bench_run_extract.py"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    sys.argv[0] = script_path
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
