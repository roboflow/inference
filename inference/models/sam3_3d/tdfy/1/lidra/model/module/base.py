from typing import Dict, Union, Optional, List, Any
from dataclasses import dataclass
import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary import (
    LayerSummary,
    get_human_readable_count,
)
from lidra.optimizer.composite import Composite as CompositeOptimizer
from lidra.scheduler.composite import Composite as CompositeScheduler
from lidra.model.io import load_model_from_checkpoint
import torch
from loguru import logger
import inspect
from easydict import EasyDict
from lidra.config.model import freeze
import peft
from lidra.callback.tdfy.compute_metrics import MetricsCallback
import lidra.mixhavior as mixhavior
from lidra.mixhavior.pytorch import PyTorchMixhaviorStateDictHandling


class ObjectOrCallable:
    def __init__(self, object_or_callable, valid_object_types) -> None:
        self._object_or_callable = object_or_callable
        self._valid_object_types = (
            valid_object_types
            if type(valid_object_types) is type
            else tuple(valid_object_types)
        )

    def get(self):
        if isinstance(self._object_or_callable, self._valid_object_types):
            return self._object_or_callable
        if callable(self._object_or_callable):
            return self._object_or_callable()
        raise RuntimeError(
            "the provided backbone is neither a pytorch module, nor a callable"
        )


class BackboneWithBenefits:
    def __init__(
        self,
        backbone,
        optimizer=None,
        scheduler=None,
        scheduler_kwargs=None,
    ) -> None:
        self.backbone = ObjectOrCallable(backbone, valid_object_types=torch.nn.Module)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

    def configure_optimizer(self, configured_backbone):
        if self.optimizer is None:
            return None
        entry = {"optimizer": self.optimizer(configured_backbone.parameters())}
        if self.scheduler is not None:
            entry["lr_scheduler"] = {
                "scheduler": self.scheduler(optimizer=entry["optimizer"])
            }
            if self.scheduler_kwargs is not None:
                entry["lr_scheduler"].update(self.scheduler_kwargs)
        return entry

    def configure_backbone(self):
        return self.backbone.get()

    def post_checkpoint_load_wrap(self, backbone):
        return backbone


class TrainableBackbone(BackboneWithBenefits):
    def __init__(
        self,
        backbone,
        optimizer,
        scheduler=None,
        scheduler_kwargs=None,
    ) -> None:
        super().__init__(
            backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
        )


class FrozenBackbone(BackboneWithBenefits):
    def __init__(
        self,
        backbone,
        optimizer=None,
        scheduler=None,
        scheduler_kwargs=None,
    ) -> None:
        if optimizer is not None:
            logger.warning(
                f"optimizer `{optimizer}` has been provided to `FrozenBackbone`, this optimizer will not be used"
            )

        if scheduler is not None:
            logger.warning(
                f"scheduler `{scheduler}` has been provided to `FrozenBackbone`, this scheduler will not be used"
            )

        super().__init__(
            backbone,
            optimizer=None,
            scheduler=None,
            scheduler_kwargs=None,
        )

    def configure_backbone(self):
        backbone = super().configure_backbone()
        return freeze(backbone)


class LoraBackbone(BackboneWithBenefits):
    def __init__(
        self,
        backbone,
        lora_config,
        optimizer,
        scheduler=None,
        scheduler_kwargs=None,
    ) -> None:
        super().__init__(backbone, optimizer, scheduler, scheduler_kwargs)
        self._lora_config = lora_config

    def post_checkpoint_load_wrap(self, backbone):
        return peft.get_peft_model(backbone, **self._lora_config)


class Base(pl.LightningModule):
    @dataclass
    class Args:
        args: List[Any]
        kwargs: Dict[str, Any]

        def with_signature(self, signature: inspect.Signature):
            arguments = signature.bind(*self.args, **self.kwargs).arguments
            return EasyDict(arguments)

    def __init__(
        self,
        models: Union[BackboneWithBenefits, Dict[str, BackboneWithBenefits]],
        checkpoint: Optional[str] = None,
        compute_metrics_callback: Optional[MetricsCallback] = None,
        disable_mixhavior_state_dict: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._base_init_models(models)
        self._checkpoint = checkpoint

        self._base_args = {}
        self.metrics_callback = compute_metrics_callback

        if not disable_mixhavior_state_dict:
            mixhavior.get_mixhavior(self).equip(PyTorchMixhaviorStateDictHandling())

    def _base_init_models(self, models):
        if isinstance(models, BackboneWithBenefits):
            models = {"main": models}

        self._base_trainable_models = models
        self._base_models = None  # to be initialized in `configure_model`

    def _check_model_is_configured(self):
        if self._base_models is None:
            raise RuntimeError(
                "base models have not been initialized, please call `module.configure_model()` before accessing base models"
            )

    @property
    def base_model(self):
        self._check_model_is_configured()
        return self._base_models["main"]

    @property
    def base_models(self):
        self._check_model_is_configured()
        return self._base_models

    def configure_optimizers(self):
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        if self._base_models is None:
            raise RuntimeError(
                "the base models have not been initialized, please call `configure_model(...)` prior to `configure_optimizers(...)`"
            )

        optimizers_and_schedulers = {
            key: opt_sche
            for key, tm in self._base_trainable_models.items()
            if (opt_sche := tm.configure_optimizer(self.base_models[key])) is not None
        }

        self._base_scheduler_names = tuple(
            key
            for key, opt_sch_dict in optimizers_and_schedulers.items()
            if ("lr_scheduler" in opt_sch_dict)
        )

        # return tuple(optimizers_and_schedulers.values())
        if len(optimizers_and_schedulers) > 0:
            optimizer = CompositeOptimizer(
                *[opt["optimizer"] for opt in optimizers_and_schedulers.values()]
            )
            if len(self._base_scheduler_names) > 0:
                scheduler = CompositeScheduler(
                    optimizer,
                    *[
                        optimizers_and_schedulers[key]["lr_scheduler"]["scheduler"]
                        for key in self._base_scheduler_names
                    ],
                )
                kwargs = [
                    {
                        k: v
                        for k, v in optimizers_and_schedulers[key][
                            "lr_scheduler"
                        ].items()
                        if k != "scheduler"
                    }
                    for key in self._base_scheduler_names
                ]
                for kwarg in kwargs:
                    assert kwarg == kwargs[0]  # should all be the same
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, **kwargs[0]},
                }
            return optimizer
        return None

    def configure_model(self) -> None:
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.hooks.ModelHooks.html#lightning.pytorch.core.hooks.ModelHooks.configure_model
        if self._base_models is None:

            def log_backbone_size(key, backbone):
                n_parameters = LayerSummary(backbone).num_parameters
                logger.debug(
                    f'model size of "{key}" is : {get_human_readable_count(n_parameters)}'
                )
                return backbone

            self._base_models = torch.nn.ModuleDict(
                {
                    key: log_backbone_size(key, tm.configure_backbone())
                    for key, tm in self._base_trainable_models.items()
                }
            )

            if self._checkpoint is not None:
                load_model_from_checkpoint(
                    self,
                    self._checkpoint,
                    strict=True,
                    device="cpu",
                    freeze=False,
                    eval=False,
                )

            # optional wrapping after loading the weights
            for key, tm in self._base_trainable_models.items():
                self._base_models[key] = tm.post_checkpoint_load_wrap(
                    self._base_models[key]
                )

    def _compute_and_log_val_metrics(self, validation_step_output, batch):
        if self.metrics_callback is None:
            return None

        metrics_df = self.metrics_callback.compute_metrics_per_sample(
            self, validation_step_output, batch
        )
        batch_size = len(batch[0])
        metrics_dict = dict(metrics_df.mean(skipna=True))
        metrics_dict = {f"val/metrics/{k}": v for k, v in metrics_dict.items()}
        self.log_dict(
            metrics_dict,
            sync_dist=True,
            batch_size=batch_size,
        )
        return metrics_dict

    def remove_args(self, name):
        del self._base_args[name]

    def add_args(self, name, args, kwargs):
        if name in self._base_args:
            raise RuntimeError(f"args named {name} is already present in base module")
        self._base_args[name] = Base.Args(args=args, kwargs=kwargs)

    def get_args(self, name, key=None):
        args = self._base_args.get(name, Base.Args(args=(), kwargs={}))
        if key is None:
            return args
        if isinstance(key, int):
            return args.args[key]
        if isinstance(key, str):
            return args.kwargs.get(key)

    def with_args(self, name, args, kwargs):
        base = self

        class WithArgs:
            def __enter__(self):
                base.add_args(name, args, kwargs)

            def __exit__(self, exc_type, exc_value, traceback):
                base.remove_args(name)

        return WithArgs()
