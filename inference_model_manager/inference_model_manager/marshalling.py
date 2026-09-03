import dataclasses
import io
import pickle
from typing import Any

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


def model_supports_rle(model: Any) -> bool:
    """True if the model's instance masks can be requested in RLE format."""
    return "rle" in getattr(model, "supported_mask_formats", set())
