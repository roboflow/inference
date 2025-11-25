from typing import Iterable, Callable, Optional
import torch
from collections import namedtuple
from loguru import logger

from lidra.data.utils import to_device

OptimizationState = namedtuple(
    "OptimizationState",
    ["idx", "model", "loss", "x", "y", "target", "optimizer"],
)


def optimize(
    dataloader: Iterable,
    input_extractor_fn: Callable,
    target_extractor_fn: Callable,
    model: torch.nn.Module,
    optimizer_fn: Callable,
    loss_fn: Callable,
    max_iteration: Optional[int] = None,
    device: Optional[str] = None,
):
    # prepare model
    model.train()
    if device is not None:
        model = model.to(device)

    # create optimizer
    optimizer = optimizer_fn(model.parameters())

    # optimization loop
    for idx, batch in enumerate(dataloader):
        if batch is None:
            logger.warning(f"batch found to be 'None' as step #{idx}")
            break

        optimizer.zero_grad()

        # input and target extraction
        x = input_extractor_fn(batch)
        target = target_extractor_fn(batch)
        if device is not None:
            x = to_device(x, device)
            target = to_device(target, device)

        # run model
        y = model(*x[0], **x[1])

        # compute loss and run backward pass
        loss = loss_fn(y, *target[0], **target[1])
        loss.backward()
        optimizer.step()

        # yield state to inform caller
        yield OptimizationState(
            idx=idx,
            model=model,
            loss=loss.detach(),
            x=x,
            y=y,
            target=target,
            optimizer=optimizer,
        )

        # stop if we overdo it
        if (max_iteration is not None) and (idx >= max_iteration - 1):
            break
