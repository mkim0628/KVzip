# ------------------------------------------------------------------------------
# Pure-Python / PyTorch fallback for the tiny_api_cuda CUDA extension.
#
# This module is used automatically when the compiled CUDA extension
# (built from csrc/) is NOT available (e.g. before running `make install`).
#
# To build the fast CUDA version:
#   cd csrc && python build.py install
# After that the compiled .so will take precedence over this stub.
#
# Functional difference: this fallback is correct but slower than the
# CUDA kernel because it operates on CPU tensors in Python loops.
# EvictCache decode-time performance will be lower, but correctness is
# preserved for testing / CI.
# ------------------------------------------------------------------------------

import warnings
import torch

warnings.warn(
    "tiny_api_cuda CUDA extension not found – using pure-Python fallback.\n"
    "For full performance, build the CUDA extension:\n"
    "    cd csrc && python build.py install",
    ImportWarning,
    stacklevel=2,
)


def update_flatten_view(
    cache: torch.Tensor,
    state: torch.Tensor,
    headlens: torch.Tensor,
    cu_headlens: torch.Tensor,
) -> torch.Tensor:
    """
    Pure-Python equivalent of the update_flatten_view CUDA kernel.

    Inserts ``t`` new rows per head into a flattened KV cache.

    Args:
        cache:       Flattened KV cache, shape (total_rows, dim).
                     total_rows = sum(headlens[h] for h in range(head_num)).
        state:       New KV states to append, shape (head_num * t, dim).
        headlens:    Current per-head lengths, shape (head_num,), dtype int32.
        cu_headlens: Cumulative per-head lengths (exclusive prefix sum),
                     shape (head_num + 1,), dtype int32.

    Returns:
        New cache tensor of shape (total_rows + head_num * t, dim) where
        for each head h the ``t`` new rows are appended right after the
        existing ``headlens[h]`` rows.
    """
    head_num = headlens.size(0)
    dim = cache.size(1)
    total_state_rows = state.size(0)
    assert total_state_rows % head_num == 0, (
        f"state rows ({total_state_rows}) must be divisible by head_num ({head_num})"
    )
    t = total_state_rows // head_num

    segments = []
    for h in range(head_num):
        start = cu_headlens[h].item()
        length = headlens[h].item()
        # existing rows for this head
        segments.append(cache[start: start + length])
        # new rows to insert
        segments.append(state[h * t: (h + 1) * t])

    return torch.cat(segments, dim=0)
