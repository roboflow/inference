import torch
import gc

from lightning.pytorch.callbacks import Callback


class GarbageCollection(Callback):
    def __init__(
        self,
        frequency,
        collect_on_epoch_end: bool = True,
        collect_on_setup: bool = False,
        collect_cuda: bool = True,
    ):
        super().__init__()
        self.frequency = frequency
        self.collect_on_epoch_end = collect_on_epoch_end
        self.collect_on_setup = collect_on_setup
        self.collect_cuda = collect_cuda

    def _garbage_collect(self):
        gc.collect()
        if self.collect_cuda:
            torch.cuda.empty_cache()

    def on_epoch_end(self, trainer, pl_module):
        if self.collect_on_epoch_end:
            self._garbage_collect()

    def _on_batch_end(self, batch_idx: int):
        if batch_idx % self.frequency == 0:
            self._garbage_collect()

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        self._on_batch_end(batch_idx)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._on_batch_end(batch_idx)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._on_batch_end(batch_idx)

    def setup(self, trainer, pl_module, stage):
        if self.collect_on_setup:
            self._garbage_collect()
