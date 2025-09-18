from typing import Generator, Tuple

import torch


def generate_batch_chunks(
    input_batch: torch.Tensor,
    chunk_size: int,
) -> Generator[Tuple[torch.Tensor, int], None, None]:
    n = input_batch.shape[0]
    for i in range(0, n, chunk_size):
        chunk = input_batch[i : i + chunk_size]
        padding_size = chunk_size - chunk.shape[0]
        if padding_size > 0:
            padding_shape = (padding_size,) + chunk.shape[1:]
            padding = torch.zeros(
                padding_shape, device=input_batch.device, dtype=input_batch.dtype
            )
            chunk = torch.cat([chunk, padding], dim=0)
        yield chunk, padding_size
