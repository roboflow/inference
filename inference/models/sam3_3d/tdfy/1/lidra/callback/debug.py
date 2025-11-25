from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger


class Debug(Callback):
    @rank_zero_only
    def setup(self, trainer, pl_module, stage: str):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def teardown(self, trainer, pl_module, stage: str):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_sanity_check_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_sanity_check_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx: int,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_test_epoch_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_predict_epoch_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_predict_epoch_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_validation_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_test_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_predict_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_predict_start(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_predict_end(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_exception(
        self,
        trainer,
        pl_module,
        exception: BaseException,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def state_dict(self):
        return {}

    @rank_zero_only
    def load_state_dict(self, state_dict):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_save_checkpoint(
        self,
        trainer,
        pl_module,
        checkpoint,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_load_checkpoint(
        self,
        trainer,
        pl_module,
        checkpoint,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_before_backward(self, trainer, pl_module, loss):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_after_backward(self, trainer, pl_module):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_before_optimizer_step(
        self,
        trainer,
        pl_module,
        optimizer,
    ):
        logger.debug("<<debug><callback>>")

    @rank_zero_only
    def on_before_zero_grad(
        self,
        trainer,
        pl_module,
        optimizer,
    ):
        logger.debug("<<debug><callback>>")
