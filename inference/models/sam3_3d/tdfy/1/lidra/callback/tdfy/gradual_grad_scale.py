import optree
import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Any
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


# From Guided Diffusion
class GradualGradScaleCallback(Callback):
    """
    Gradual gradient scaling from OAI Guided Diffusion

    Doubles as way to skip batches if the gradients are NaN or Inf.
    """

    def __init__(
        self,
        initial_scale: float = 20.0,
        scale_growth: float = 0.001,
        scale_cut: float = 1.0,
    ):
        super().__init__()
        self.current_scale = initial_scale
        self.scale_growth = scale_growth
        # skip the first zero_grad
        self.found_infinite = True
        self.model_params = []
        self.master_params = []
        self.device = None
        self.ckpt_master_params = None
        self.scale_cut = scale_cut

    @staticmethod
    def is_leaf_param(x):
        return isinstance(x, list) and x and isinstance(x[0], nn.Parameter)

    def _make_master_params(self):

        def _make_master_param_group(group_param):
            group_master_param = _flatten_dense_tensors(
                [param.detach().float() for param in group_param]
            )
            group_master_param = nn.Parameter(group_master_param)
            group_master_param.requires_grad = True
            return [group_master_param]  # keep leaf as list of tensors

        self.master_params = optree.tree_map(
            _make_master_param_group,
            self.model_params,
            is_leaf=self.is_leaf_param,
        )

    def _model_grads_to_master_grads(self):
        """
        Copy the gradients from the model parameters into the master parameters
        from make_master_params().
        """

        def _set_master_grad(group_params, master_param):
            flat_grad = _flatten_dense_tensors(
                [p.grad.detach().float() for p in group_params]
            )
            master_param[0].grad = flat_grad

        optree.tree_map(
            _set_master_grad,
            self.model_params,
            self.master_params,
            is_leaf=self.is_leaf_param,
        )

    def _master_params_to_model_params(self):
        """
        Copy the master parameter data back into the model parameters.
        """

        def _cp_master_param_to_model(group_params, master_param):
            for param, master_param in zip(
                group_params,
                _unflatten_dense_tensors(master_param[0].detach(), group_params),
            ):
                param.detach().copy_(master_param)

        optree.tree_map(
            _cp_master_param_to_model,
            self.model_params,
            self.master_params,
            is_leaf=self.is_leaf_param,
        )

    def _build_params(self, trainer):
        for opt in trainer.optimizers:
            opt_params = []
            for group in opt.param_groups:
                group_params = []
                for param in group["params"]:
                    if param.requires_grad:
                        group_params.append(param)
                opt_params.append(group_params)
            self.model_params.append(opt_params)

    def on_fit_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ):
        # clear the old ones at initialization
        self.device = pl_module.device
        self.model_params.clear()
        self.master_params.clear()
        self._build_params(trainer)
        self._make_master_params()

        for i, opt in enumerate(trainer.optimizers):
            for j, group in enumerate(opt.param_groups):
                group["params"] = self.master_params[i][j]

        if self.ckpt_master_params is not None:
            for i in range(len(self.master_params)):
                for j in range(len(self.master_params[i])):
                    self.master_params[i][j][0].data.copy_(
                        self.ckpt_master_params[i][j].to(self.device)
                    )
            self.ckpt_master_params = None

    def on_before_backward(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        loss: "torch.Tensor",
    ):
        loss *= 2**self.current_scale

    def on_after_backward(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ):
        self.found_infinite = False

        # Check every parameter in the optimizer
        for opt_params in self.model_params:
            for param_group in opt_params:
                for param in param_group:
                    if param.grad is not None:
                        # If any entry in p.grad is not finite (Inf or NaN), mark it
                        if not torch.isfinite(param.grad).all():
                            self.found_infinite = True
                            break
                    if self.found_infinite:
                        break
        if self.found_infinite:
            # We found NaN or Inf in the gradients, so skip this step.
            # A common way is to zero or set the gradients to None.
            for opt_params in self.master_params:
                for master_params in opt_params:
                    master_params[0].grad = None

            # Optionally log a message or use trainer.logger
            trainer.print(
                f"[Warning] Found NaN/Inf gradients. Skipping step for optimizer."
            )
            self.current_scale -= self.scale_cut
            trainer.print(f"adjusted scale now {self.current_scale}")
        else:
            self._model_grads_to_master_grads()
            for opt_params in self.master_params:
                for master_params in opt_params:
                    master_params[0].grad.mul_(1.0 / (2**self.current_scale))

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ):
        def disable_model_param_grad(param_group):
            for param in param_group:
                if param.grad is not None:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()

        optree.tree_map(
            disable_model_param_grad,
            self.model_params,
            is_leaf=self.is_leaf_param,
        )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        if not self.found_infinite:
            self._master_params_to_model_params()
            self.current_scale += self.scale_growth

    def state_dict(self):
        def detach_param(master_param):
            return master_param[0].detach().cpu()

        return {
            "current_scale": self.current_scale,
            "scale_growth": self.scale_growth,
            "master_params": optree.tree_map(
                detach_param,
                self.master_params,
                is_leaf=self.is_leaf_param,
            ),
        }

    def load_state_dict(self, state_dict: dict):
        self.current_scale = state_dict.get("current_scale", self.current_scale)
        self.scale_growth = state_dict.get("scale_growth", self.scale_growth)
        # defer the loading to fit start, since master_params are not initialized
        self.ckpt_master_params = state_dict.get(
            "master_params", self.ckpt_master_params
        )
