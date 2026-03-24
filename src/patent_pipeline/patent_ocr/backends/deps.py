from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Sequence, Set

DOCTR_DEPS = ["python-doctr[torch] @ git+https://github.com/mindee/doctr.git"]
LIGHTONOCR_DEPS = [
    "transformers>=5.0.0",
    "accelerate>=1.0.0,<2.0.0",
    "safetensors",
    "pypdfium2",
]

_INSTALLED_BACKENDS: Set[str] = set()


def _pip_install(deps: Sequence[str]) -> None:
    if not deps:
        return
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", *deps]
    subprocess.check_call(cmd)


def ensure_backend_deps(backend_name: str, deps: Sequence[str], auto_install: bool) -> None:
    if not auto_install or not deps or backend_name in _INSTALLED_BACKENDS:
        return
    print(f"[backend-deps] Installing {backend_name} dependencies: {deps}")
    _pip_install(deps)
    _INSTALLED_BACKENDS.add(backend_name)


def import_module_with_auto_install(
    module_name: str,
    backend_name: str,
    deps: Sequence[str],
    auto_install: bool,
    err_hint: str,
) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        ensure_backend_deps(backend_name, deps, auto_install)
        try:
            return importlib.import_module(module_name)
        except ImportError:
            install_cmd = "python -m pip install " + " ".join(deps)
            raise RuntimeError(
                f"{err_hint}\nInstall via: {install_cmd}"
            ) from exc
