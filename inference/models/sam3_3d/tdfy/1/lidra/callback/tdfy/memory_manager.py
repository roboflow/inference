import math
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Any, Callable
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


def get_input_size_slat_flow(batch):
    """
    Calculate total input size for sparse tensors
    """
    all_sparse_t = batch[1]["sparse_t"]
    size = 0
    for sparse_t in all_sparse_t:
        size += sparse_t.mean.shape[0]
    return size


def set_grad_ckpt_slat(trainer, model, batch, slat_path, mem_ratio=1.0):
    """
    Used specifically for sparse latent features, when the input has varying sizes.
    Set gradient checkpointing to save memory based on predicted memory usage.
    """
    if mem_ratio == 1.0:
        return 1.0, None
    target_model = rgetattr(model, slat_path)
    num_blocks = len(target_model.blocks)
    num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
    exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
    for i in range(num_blocks):
        target_model.blocks[i].use_checkpoint = i < num_checkpoint_blocks

    def unset_func():
        for i in range(num_blocks):
            target_model.blocks[i].use_checkpoint = False

    return exact_mem_ratio, unset_func


def rgetattr(obj, dotted_path=None):
    """
    Recursively get attributes to localize model
    """
    if dotted_path is None:
        return obj
    for attr in dotted_path.split("."):
        if type(obj) == dict:
            obj = obj[attr]
        else:
            obj = getattr(obj, attr)
    return obj


# From Trellis
class MemoryManagerCallback(Callback):
    """
    A callback that performs operations based on predicted memory usage.
    The callback fits a linear model based on true memory usage and input size.
    Args:
        get_size_func: a function that takes in batch and outputs the size of the input.
        mem_based_func: a function that performs some operations based on the mem_ratio
            predictions from the linear model. Takes input of trainer, model, batch.
            Outputs the updated mem_ratio adjusted by the operation (used to fit linear model
            with actual memory usage) and an unset_func that will be called end of this trianing
            batch.
        buffer_size: the buffer size that holds memory usage. Information used to fit linear
            model.
        update_every: update frequency for the linear model and max_mem_ratio.
        target_ratio: target memory usage ratio upper bound
        max_mem_ratio_start: starting point for maximum memory ratio usage. Start from small value
            to avoid memory OOM when linear model is weak. Updated based on update_every.
        device: by default will read current device. Gave argument for cpu use case (testing).
        available_memory: by default read from device. Gave argument for cpu use case (testing).
    """

    def __init__(
        self,
        get_size_func: Callable,
        mem_based_func: Callable,
        buffer_size=1000,
        update_every=500,
        target_ratio=0.8,
        max_mem_ratio_start=0.1,
        # give these two option for testing on cpu
        device=None,
        available_memory=None,
    ):
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.target_ratio = target_ratio
        self.device = device if device is not None else torch.cuda.current_device()
        self.available_memory = (
            available_memory
            if available_memory is not None
            else torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        )

        self._memory = np.zeros(buffer_size, dtype=np.float32)
        self._input_size = np.zeros(buffer_size, dtype=np.float32)
        self._mem_ratio = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_ptr = 0
        self._buffer_length = 0

        # param for fitting memory use predictor
        self._params = (0.0, 0.0)
        self._max_mem_ratio = max_mem_ratio_start
        self.step = 0

        self._last_input_size = None
        self._last_mem_ratio = []

        self.get_size_func = get_size_func
        self.mem_based_func = mem_based_func

    def _add_sample(self, memory, input_size, mem_ratio):
        self._memory[self._buffer_ptr] = memory
        self._input_size[self._buffer_ptr] = input_size
        self._mem_ratio[self._buffer_ptr] = mem_ratio
        self._buffer_ptr = (self._buffer_ptr + 1) % self.buffer_size
        self._buffer_length = min(self._buffer_length + 1, self.buffer_size)

    def _fit_params(self):
        memory_usage = self._memory[: self._buffer_length]
        input_size = self._input_size[: self._buffer_length]
        mem_ratio = self._mem_ratio[: self._buffer_length]

        x = input_size * mem_ratio
        y = memory_usage
        k, b = np.polyfit(x, y, 1)
        self._params = (k, b)

    def _get_mem_ratio(self, input_size):
        k, b = self._params
        if k == 0:
            return np.random.rand() * self._max_mem_ratio
        pred = (self.available_memory * self.target_ratio - b) / (k * input_size)
        return min(self._max_mem_ratio, max(0.0, pred))

    def _update_run_states(self, input_size=None, mem_ratio=None):
        if self._last_input_size is None:
            self._last_input_size = input_size
        # inherit from Trellis; not really useful here.
        elif self._last_input_size != input_size:
            raise ValueError(
                f"Input size should not change for different ElasticModules."
            )
        self._last_mem_ratio.append(mem_ratio)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ):
        # record memory use
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._last_input_size = None
        self._last_mem_ratio = []
        input_size = self.get_size_func(batch)
        mem_ratio = self._get_mem_ratio(input_size)
        # perform memory based operations
        exact_mem_ratio, self.unset_func = self.mem_based_func(
            trainer,
            pl_module,
            batch,
            mem_ratio=mem_ratio,
        )
        self._update_run_states(input_size, exact_mem_ratio)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        if self.unset_func is not None:
            self.unset_func()
        # otherwise we can manually set memory for cpu testing
        if torch.cuda.is_available() and self.device != "cpu":
            self._last_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3
        # Not sure if this is necessary; we do it on every device I believe
        # Maybe we can sync across machines?
        self._last_mem_ratio = sum(self._last_mem_ratio) / len(self._last_mem_ratio)
        self._add_sample(self._last_memory, self._last_input_size, self._last_mem_ratio)
        self.step += 1
        # increate memory allowance once model pred becomes better
        if self.step % self.update_every == 0:
            self._max_mem_ratio = min(1.0, self._max_mem_ratio + 0.1)
            self._fit_params()

    def state_dict(self):
        return {
            "_params": self._params,
        }

    def load_state_dict(self, state_dict: dict):
        self._params = tuple(state_dict["_params"])
