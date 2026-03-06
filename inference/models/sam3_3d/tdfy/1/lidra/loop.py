from collections.abc import Mapping
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Union
from loguru import logger
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from lidra.model.module.base import Base as ModuleBase

from lightning.pytorch.loggers import WandbLogger


class Loop(pl.Trainer):
    def __init__(
        self,
        module: ModuleBase,
        dataloaders: Union[pl.LightningDataModule, Dict[str, DataLoader]],
        checkpoint: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._loop_module = module
        self._init_checkpoint(checkpoint)
        self._base_init_dataloaders(dataloaders)

    def _init_checkpoint(self, checkpoint):
        last_checkpoints = self.checkpoint_callback._find_last_checkpoints(self)
        if len(last_checkpoints) > 0:
            logger.info(f"found existing checkpoints in folder, will resume from it")
            if checkpoint is not None:
                logger.warning(
                    "overriding existing checkpoint to use the latest ckpt in the folder"
                )
            checkpoint = "last"  # auto-resume
        self._checkpoint = checkpoint

    def log_config(self, config: DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        for logger_obj in self.loggers:
            if hasattr(logger_obj, "log_hyperparams"):
                logger.info(f"Logging config to {type(logger_obj).__name__}")
                logger_obj.log_hyperparams(config_dict)

        # Validate wandb tags length
        if WandbLogger is not None:
            for logger_obj in self.loggers:
                if isinstance(logger_obj, WandbLogger):
                    tags = getattr(logger_obj, "tags", None) or []
                    valid_tags = []
                    for tag in tags:
                        tag_str = str(tag) if tag is not None else ""
                        if len(tag_str) > 64:
                            logger.warning(
                                f"Wandb tag '{tag_str}' exceeds 64 characters ({len(tag_str)} chars), ignoring this tag"
                            )
                        else:
                            valid_tags.append(tag)

                    # Update the logger's tags to only include valid ones
                    if len(valid_tags) != len(tags):
                        logger_obj.tags = valid_tags
                        logger.info(
                            f"Updated wandb tags to only include {len(valid_tags)} valid tags (out of {len(tags)} original)"
                        )

    @property
    def module(self):
        return self._loop_module

    def _base_init_dataloaders(self, dataloaders):
        self._loop_datamodule = None
        self._loop_dataloader_training = None
        self._loop_dataloader_validation = None
        self._loop_dataloader_test = None

        if isinstance(dataloaders, pl.LightningDataModule):
            self._loop_datamodule = dataloaders
        elif isinstance(dataloaders, Mapping):
            valid_keys = {"training", "validation", "test"}
            for key, data in dataloaders.items():
                if key in valid_keys:
                    if isinstance(data, DataLoader):
                        setattr(self, f"_loop_dataloader_{key}", data)
                    elif data is None:
                        pass
                    else:
                        raise RuntimeError(
                            f"invalid type '{type(data)}' has been found in the dataloaders dictionary (key='{key}'), valid type should subclass the pytorch 'DataLoader' class"
                        )
                else:
                    raise RuntimeError(
                        f"invalid key '{key}' has been found in the dataloaders dictionary, valid keys are in the set {valid_keys}"
                    )
        elif dataloaders is None:
            logger.warning(
                "No dataloaders have been provided. fit(), test(), predict() will not be available."
            )
        else:
            raise RuntimeError(
                f"invalid dataloaders type '{type(dataloaders)}' has been found, valid types are 'LightningDataModule' or a dictionary of 'DataLoader'"
            )

    def fit(self, *args, **kwargs):
        with self._loop_module.with_args("fit", args, kwargs):
            return pl.Trainer.fit(
                self,
                model=self._loop_module,
                train_dataloaders=self._loop_dataloader_training,
                val_dataloaders=self._loop_dataloader_validation,
                datamodule=self._loop_datamodule,
                ckpt_path=self._checkpoint,
            )

    def test(self, *args, dataloader=None, **kwargs):
        dataloader = self._loop_dataloader_test if dataloader is None else dataloader
        with self._loop_module.with_args("test", args, kwargs):
            return pl.Trainer.test(
                self,
                model=self._loop_module,
                dataloaders=dataloader,
                datamodule=self._loop_datamodule,
            )

    def predict(self, *args, dataloader=None, **kwargs):
        dataloader = self._loop_dataloader_test if dataloader is None else dataloader
        with self._loop_module.with_args("predict", args, kwargs):
            return pl.Trainer.predict(
                self,
                model=self._loop_module,
                dataloaders=dataloader,
                datamodule=self._loop_datamodule,
                return_predictions=True,
            )
