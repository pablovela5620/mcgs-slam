"""Native-torch replacements for the two torch_scatter ops used in this repo.

torch_scatter ships no wheels for torch 2.13 + cu130 (and none for
linux-aarch64), so scatter_sum / scatter_mean are implemented here with
torch.Tensor.scatter_add_, which is the same atomic-add kernel torch_scatter
uses for these reductions. Semantics match torch_scatter for a 1-D index
broadcast along `dim`, which is the only pattern used in this codebase.
"""

import torch
from torch import Tensor


def _expand_index(index: Tensor, src: Tensor, dim: int) -> Tensor:
    """Broadcast a 1-D index of length src.shape[dim] to src's full shape.

    Args:
        index: Int64[Tensor, "n"] scatter target indices.
        src: source tensor with src.shape[dim] == n.
        dim: dimension along which to scatter.
    """
    if index.dim() != 1:
        raise ValueError(f"only 1-D indices are supported, got {index.dim()}-D")
    view_shape: list[int] = [1] * src.dim()
    view_shape[dim] = -1
    return index.view(view_shape).expand_as(src)


def scatter_sum(src: Tensor, index: Tensor, dim: int, dim_size: int | None = None) -> Tensor:
    """Sum-reduce src into dim_size bins along dim, like torch_scatter.scatter_sum."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    out_shape: list[int] = list(src.shape)
    out_shape[dim] = dim_size
    out: Tensor = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, _expand_index(index, src, dim), src)


def scatter_mean(src: Tensor, index: Tensor, dim: int, dim_size: int | None = None) -> Tensor:
    """Mean-reduce src into dim_size bins along dim, like torch_scatter.scatter_mean."""
    total: Tensor = scatter_sum(src, index, dim, dim_size)
    # Bin counts depend only on the 1-D index; counting with ones_like(src)
    # would allocate and scatter a full copy of src for nothing.
    ones: Tensor = torch.ones(index.shape, dtype=src.dtype, device=src.device)
    count: Tensor = scatter_sum(ones, index, 0, total.shape[dim])
    view_shape: list[int] = [1] * total.dim()
    view_shape[dim] = -1
    return total / count.clamp(min=1).view(view_shape)
