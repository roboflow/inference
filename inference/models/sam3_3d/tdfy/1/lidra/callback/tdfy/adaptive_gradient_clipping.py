import torch
import numpy as np
import torch.utils
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class AdaptiveGradientClipCallback(Callback):
    """
    Clip gradient by a fixed max value or the max value of clip_percentile in a moving buffer.
    Use the max of the two to clip
    """

    def __init__(
        self,
        max_clip_val: float = 1.0,
        clip_percentile: float = 95.0,
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.max_clip_val = max_clip_val
        self.clip_percentile = clip_percentile
        self.buffer_size = buffer_size
        self.buffer_ptr = 0
        self.buffer_length = 0
        self.grad_norms = np.zeros(buffer_size, dtype=np.float32)
        self.max_norm = max_clip_val
        self.model_params = []

    # store parameters from optimizers, which can be float32 instead of fp16
    def on_train_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ):
        # clear the old ones at initialization
        # use the one from optimizer, since other plugins might change which to use
        self.model_params.clear()
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    self.model_params.append(param)

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer,
    ):
        """
        Called after every backward() call.
        Here, you have access to all gradients via `pl_module`.
        """
        # clip by max norm
        max_norm = self.max_norm if self.max_norm is not None else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model_params,
            max_norm=max_norm,
            norm_type=2.0,
            error_if_nonfinite=False,
            foreach=None,
        )

        # update max_norm
        if torch.isfinite(grad_norm):
            self.grad_norms[self.buffer_ptr] = grad_norm
            self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
            self.buffer_length = min(self.buffer_length + 1, self.buffer_size)
            if self.buffer_length == self.buffer_size:
                self.max_norm = np.percentile(self.grad_norms, self.clip_percentile)
                self.max_norm = min(self.max_clip_val, self.max_norm)

        return grad_norm
