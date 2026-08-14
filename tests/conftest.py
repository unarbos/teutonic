"""Test harness shims.

`validator.py` calls `chain_config.load_arch()` at import time, which imports
`archs.quasar`, which imports `fla` (flash-linear-attention) — a CUDA/triton
package that does not install on a CPU dev box. The gates under test here
(reveal parsing, auto_map validation) are pure and independent of the model
implementation, so we pre-register a stub under the arch module name:
`importlib.import_module` returns an existing `sys.modules` entry without
executing the real module.

This shim is only reached when the real package is unavailable, so on the GPU
hosts the genuine arch module is imported and the tests exercise the same code
path as production.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_arch_if_unavailable() -> None:
    try:
        import fla  # noqa: F401
    except ImportError:
        pass
    else:
        return  # real dependency present; let the genuine arch module load

    import chain_config

    name = chain_config.ARCH_MODULE
    if not name or name in sys.modules:
        return
    # Register parents too, so `import archs.quasar` resolves without touching
    # the real package's __init__.
    parts = name.split(".")
    for idx in range(1, len(parts) + 1):
        dotted = ".".join(parts[:idx])
        sys.modules.setdefault(dotted, types.ModuleType(dotted))


os.environ.setdefault("TEUTONIC_MODEL_CACHE_DIR", "/tmp/teutonic/test_models")
_stub_arch_if_unavailable()
