import dataclasses
import io
import pickle
from typing import Any, Callable, List, Optional

import numpy as np


def to_bytes(raw_input: Any) -> bytes:
    """Serialise any input value to bytes.

    Returns:
        bytes, bytearray, memoryview  →  bytes (zero-copy when possible)
        numpy ndarray                 →  numpy .npy bytes (magic b'\\x93NUMPY')
        anything else                 →  pickle
    """
    if isinstance(raw_input, (bytes, bytearray)):
        return bytes(raw_input)
    if isinstance(raw_input, memoryview):
        return bytes(raw_input)
    if isinstance(raw_input, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, raw_input, allow_pickle=False)
        return buf.getvalue()
    return pickle.dumps(raw_input)


# Resolved once on first use; avoids importing torch when it's never needed.
_torch = None


def tensors_to_numpy(result: Any) -> Any:
    """Convert every torch.Tensor in a result to CPU numpy, in place.

    Pickling numpy is ~10x faster than pickling torch tensors, and keeps CUDA
    tensors off the wire so the receiver needs no GPU. Walks any result shape:
    dataclass, list/tuple, dict, bare tensor; everything else passes through.
    """
    global _torch
    if _torch is None:
        import torch  # noqa: PLC0415

        _torch = torch
    Tensor = _torch.Tensor

    def _walk(obj: Any) -> Any:
        if isinstance(obj, Tensor):
            t = obj.detach()
            if t.dtype == _torch.bfloat16:
                t = t.float()
            return t.cpu().numpy()
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            # object.__setattr__ so frozen dataclasses (SAM predictions,
            # embeddings) convert too instead of raising FrozenInstanceError
            for f in dataclasses.fields(obj):
                object.__setattr__(obj, f.name, _walk(getattr(obj, f.name)))
            return obj
        if isinstance(obj, list):
            return [_walk(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(x) for x in obj)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        return obj

    return _walk(result)


def split_batched_result(
    raw_out: Any, n_images: int, retry_single: Optional[Callable] = None
) -> List[Any]:
    """Map a raw batched model result to one result per input image.

    Handles the result shapes the model contract can produce:
      * a list whose length matches the batch (or a single-image call),
      * a tuple of per-image lists (the structured-OCR ``(texts, detections)``
        contract) — each image gets the whole tuple back with one-element
        lists, exactly the shape a single-image call returns,
      * an array/tensor whose leading dimension matches the batch,
      * anything else, which falls back to ``retry_single`` — re-invoking the
        model once per image — when one is supplied.
    """
    if isinstance(raw_out, list) and (n_images == 1 or len(raw_out) == n_images):
        return raw_out
    if (
        n_images > 1
        and isinstance(raw_out, tuple)
        and raw_out
        and all(
            isinstance(element, list) and len(element) == n_images
            for element in raw_out
        )
    ):
        return [
            tuple([element[index]] for element in raw_out) for index in range(n_images)
        ]
    shape = getattr(raw_out, "shape", None)
    if shape and n_images > 1 and shape[0] == n_images:
        return [raw_out[i : i + 1] for i in range(n_images)]
    if n_images == 1:
        return [raw_out]
    if retry_single is not None:
        results = []
        for index in range(n_images):
            single_out = retry_single(index)
            if isinstance(single_out, list):
                single_out = single_out[0] if single_out else None
            results.append(single_out)
        return results
    return [raw_out]


def model_supports_rle(model: Any) -> bool:
    """True if the model's instance masks can be requested in RLE format."""
    return "rle" in getattr(model, "supported_mask_formats", set())
