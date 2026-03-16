# ------------------------------------------------------------------------------
# KVzip-vLLM Integration
# Provides a vLLM-compatible interface for KVzip KV cache compression.
# ------------------------------------------------------------------------------

# Ensure tiny_api_cuda is resolvable (compiled .so or pure-Python stub)
# before any KVzip sub-module is imported.
import sys as _sys
import os as _os

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

try:
    import tiny_api_cuda  # noqa: F401
except ModuleNotFoundError:
    import importlib.util as _ilu
    _stub_path = _os.path.join(_REPO_ROOT, "tiny_api_cuda.py")
    _spec = _ilu.spec_from_file_location("tiny_api_cuda", _stub_path)
    _stub = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_stub)
    _sys.modules["tiny_api_cuda"] = _stub

from .engine import KVzipVLLMEngine
from .types import SamplingParams, CompletionOutput, RequestOutput

__all__ = [
    "KVzipVLLMEngine",
    "SamplingParams",
    "CompletionOutput",
    "RequestOutput",
]
