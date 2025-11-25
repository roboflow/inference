import torch
from loguru import logger
from torch import Tensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)
import typing
from typing import Union, Optional


_tensor_or_tensors = Union[
    torch.Tensor,
    typing.Iterable[torch.Tensor],  # noqa: UP006 - needed until XLA's patch is updated
]


# run the following function after loss has been computed, but before the backward pass
# usefull to fix "unused parameters found" exception returned by DDP
def find_unused_parameters(module: torch.nn.Module, loss: torch.Tensor):
    loss.backward(retain_graph=True)

    for name, param in module.named_parameters():
        if param.grad is None:
            logger.warning(
                f'parameter "{name}" has not been used in the calculation of the loss'
            )

    module.zero_grad()


# this implementation is taken from a future version (2.8.0) of PyTorch:
# https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/utils/clip_grad.py#L175
@torch.no_grad
def _get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype(
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: list[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm
