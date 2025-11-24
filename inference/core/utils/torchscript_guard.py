from contextlib import contextmanager
from threading import RLock

import torch

_torch_jit_script_lock = RLock()


@contextmanager
def _temporarily_disable_torch_jit_script():
    """
    Temporarily override torch.jit.script with an identity function. This is
    useful to avoid TorchScript redefinition errors in environments where
    torchvision transforms are repeatedly scripted.
    """
    with _torch_jit_script_lock:
        original_script = torch.jit.script
        try:
            torch.jit.script = lambda module, *args, **kwargs: module  # type: ignore[assignment]
            yield
        finally:
            torch.jit.script = original_script  # type: ignore[assignment]
