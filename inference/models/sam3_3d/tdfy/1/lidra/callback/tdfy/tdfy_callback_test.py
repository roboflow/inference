import os
import unittest
import torch
import torch.nn as nn
import lightning.pytorch as pl
from functools import partial
from torch.utils.data import DataLoader, TensorDataset
from lidra.callback.tdfy.adaptive_gradient_clipping import AdaptiveGradientClipCallback
from lidra.callback.tdfy.gradual_grad_scale import GradualGradScaleCallback
from lidra.callback.tdfy.memory_manager import MemoryManagerCallback, set_grad_ckpt_slat


class DummyLinearModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.l1(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    @staticmethod
    def _loss_func(pred, y):
        return nn.functional.mse_loss(pred, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self._loss_func(pred, y)
        return loss


class DummyModelWrapper(DummyLinearModel):
    def __init__(self):
        super().__init__()
        self.submodule = DummyLinearModel()
        self.use_checkpoint = False

    def forward(self, x):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self.submodule, x, use_reentrant=False
            )
        else:
            return self.submodule(x)


class MultiDummyModel(DummyLinearModel):
    def __init__(self, num_layers=1):
        self.blocks = [DummyModelWrapper() for _ in range(num_layers)]

    def forward(self, x):
        for block in self.blocks:
            y = block(x)
        return y


class NaNDummyLinearModel(DummyLinearModel):
    # create NaN loss by division by zero
    @staticmethod
    def _loss_func(pred, y):
        return ((pred - y) ** 2 / 0).mean()


class TestCustomGradientClipCallback(unittest.TestCase):
    def test_max_val_gradient_clipping(self):
        model = DummyLinearModel()
        grad_clip_callback = AdaptiveGradientClipCallback(max_clip_val=0.1)

        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=5)

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            enable_checkpointing=False,
            callbacks=[grad_clip_callback],
            logger=False,
            enable_model_summary=False,
        )

        trainer.fit(model, train_dataloaders=loader)

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
        total_norm = total_norm**0.5

        self.assertLessEqual(
            total_norm,
            0.1 + 1e-7,
            f"Grad norm {total_norm} should have been clipped to <= 0.1",
        )

    def test_percentile_gradient_clipping(self):

        model = DummyLinearModel()
        grad_clip_callback = AdaptiveGradientClipCallback(
            max_clip_val=1.0, buffer_size=100
        )

        # set to assume buffer has been filled
        grad_clip_callback.grad_norms[:6] = 0.5
        grad_clip_callback.buffer_length = 101

        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=5)

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,  # run twice to apply percentile
            enable_checkpointing=False,
            callbacks=[grad_clip_callback],
            logger=False,
            enable_model_summary=False,
        )

        trainer.fit(model, train_dataloaders=loader)

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm**2
        total_norm = total_norm**0.5

        self.assertLessEqual(
            total_norm,
            0.5 + 1e-7,
            f"Grad norm {total_norm} should have been clipped to <= 0.5",
        )


class TestGradualGradScaleCallback(unittest.TestCase):
    def test_normal_cases(self):
        # init dummy staff
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=5)
        model = DummyLinearModel()

        # init callback, with values diff from defaults
        grad_scale_callback = GradualGradScaleCallback(
            initial_scale=5.0,
            scale_growth=0.5,
        )

        trainer = pl.Trainer(
            max_epochs=2,
            limit_train_batches=2,
            enable_checkpointing=False,
            callbacks=[grad_scale_callback],
            logger=False,
            enable_model_summary=False,
        )

        trainer.fit(model, train_dataloaders=loader)

        # After training, the scale should have increased
        final_scale = grad_scale_callback.current_scale
        self.assertAlmostEqual(
            final_scale,
            7.0,  # 5 + 4 * 0.5
            delta=1e-3,
            msg="Callback gradient scale should grow to the correct value",
        )

        # Save checkpoint
        ckpt_path = "test_grad_scale.ckpt"
        trainer.save_checkpoint(ckpt_path)

        # load from ckpt
        new_model = DummyLinearModel()
        new_callback = GradualGradScaleCallback()  # re-init with default
        new_trainer = pl.Trainer(
            max_epochs=3,
            limit_train_batches=1,
            callbacks=[new_callback],
            logger=False,
            enable_model_summary=False,
            enable_checkpointing=False,
        )
        new_trainer.fit(new_model, train_dataloaders=loader, ckpt_path=ckpt_path)

        # check if state of callback has been restored
        self.assertAlmostEqual(
            new_callback.current_scale,
            final_scale + new_callback.scale_growth,
            delta=1e-3,
            msg="Callback's scale should resume from checkpointed value and then increase again",
        )

        # Cleanup
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    def test_nan_cases(self):
        # init dummy staff
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=5)
        model = NaNDummyLinearModel()
        old_weight = model.l1.weight.detach().clone()

        # init callback, with values diff from defaults
        grad_scale_callback = GradualGradScaleCallback(
            initial_scale=5.0,
            scale_growth=0.5,
        )

        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            enable_checkpointing=False,
            callbacks=[grad_scale_callback],
            logger=False,
            enable_model_summary=False,
        )

        trainer.fit(model, train_dataloaders=loader)

        # After training, the scale should have increased
        self.assertTrue(
            grad_scale_callback.found_infinite,
            msg="Callback gradient scale should detect NaN",
        )
        self.assertAlmostEqual(
            grad_scale_callback.current_scale,
            4.0,  # minus one
            delta=1e-3,
            msg="Callback's scale should reduce by 1 when hitting NaN",
        )
        self.assertFalse(
            torch.isnan(model.l1.weight).any().item(),
            msg="Callback should avoid NaN gradient",
        )
        self.assertTrue(
            torch.equal(model.l1.weight, old_weight),
            msg="Callback should skip NaN and not modify the model weight",
        )


class TestMemoryManagerCallback(unittest.TestCase):
    # since we directly
    def testMemoryManagerCPU(self):
        model = MultiDummyModel(num_layers=10)
        mem_manager_callback = MemoryManagerCallback(
            get_size_func=lambda x: x.shape[0],
            mem_based_func=partial(set_grad_ckpt_slat, slat_path=None),
            device="cpu",
            available_memory=10,
            target_ratio=0.5,
            max_mem_ratio_start=1.0,
        )

        # assume linear relationship
        mem_manager_callback._params = (0.1, 0.0)

        # test one loop, no memory overflow
        mem_manager_callback.on_train_batch_start(
            trainer=None,  # placeholder
            pl_module=model,
            batch=torch.randn(16, 10),
            batch_idx=0,
        )
        self.assertAlmostEqual(mem_manager_callback._last_mem_ratio[-1], 1.0)
        # set to memorize usage
        mem_manager_callback._last_memory = 1.6
        mem_manager_callback.on_train_batch_end(
            trainer=None,
            pl_module=model,
            outputs=None,
            batch=None,
            batch_idx=None,
        )

        # test one loop, with memory overflow
        mem_manager_callback.on_train_batch_start(
            trainer=None,  # placeholder
            pl_module=model,
            batch=torch.randn(100, 10),
            batch_idx=0,
        )
        self.assertAlmostEqual(mem_manager_callback._last_mem_ratio[-1], 0.5)
        for i in range(6):
            self.assertTrue(model.blocks[i].use_checkpoint)
        for i in range(4):
            self.assertFalse(model.blocks[i + 6].use_checkpoint)
        mem_manager_callback._last_memory = 5
        mem_manager_callback.on_train_batch_end(
            trainer=None,
            pl_module=model,
            outputs=None,
            batch=None,
            batch_idx=None,
        )
        for i in range(10):
            self.assertFalse(model.blocks[i].use_checkpoint)

        # test parameter fitting
        for j in range(4):
            j += 1
            # test one loop, with memory overflow
            mem_manager_callback.on_train_batch_start(
                trainer=None,  # placeholder
                pl_module=model,
                batch=torch.randn(10 * j, 10),
                batch_idx=0,
            )
            mem_manager_callback._last_memory = j
            mem_manager_callback.on_train_batch_end(
                trainer=None,
                pl_module=model,
                outputs=None,
                batch=None,
                batch_idx=None,
            )
        mem_manager_callback._params = (0.2, 0.0)
        mem_manager_callback._fit_params()
        self.assertAlmostEqual(mem_manager_callback._params[0], 0.1)
        self.assertAlmostEqual(mem_manager_callback._params[1], 0.0)


if __name__ == "__main__":
    unittest.main()
