import unittest
import torch

from lidra.model.backbone.generator.flow_matching.model import (
    FlowMatching,
    uniform_sampler,
)
from lidra.test.util import run_unittest
import optree
from loguru import logger
from unittest.mock import Mock


class UnitTests(unittest.TestCase):
    # TODO(Pierre) : write better unit tests when generator's interface is finalized

    def _test_flow_matching(self, flow_matching: FlowMatching, x, device="cpu"):
        x_shape = optree.tree_map(lambda x: x.shape, x)
        loss, detail_losses = flow_matching.loss(x)

        self.assertEqual(loss.shape, ())  # loss is a scalar
        self.assertEqual(type(detail_losses), dict)

        gen_x = flow_matching.generate(x_shape, x_device=device)

        gen_x_shape = optree.tree_map(lambda x: x.shape, gen_x)
        self.assertEqual(x_shape, gen_x_shape)

        llh = flow_matching.log_likelihood(gen_x, z_samples=2)

    def test_flow_matching_broadcasting_loss(self):
        batch_size = 4
        x = torch.ones(size=(batch_size, 32, 32)) * 3.14
        x[1] += 1.0
        x[2] += 2.0
        x[3] += 3.0
        reverse_fn = lambda x, _: x

        def custom_loss_0(pred, target):
            # should return 4.0
            return (pred[:, 0, 0] - target[:, 0, 0] + 2.0).sum()

        def custom_loss_1(pred, target):
            # should return 8.0
            return (pred[:, 0, 0] - target[:, 0, 0] + 3.0).sum()

        loss_fn = {
            "translation": custom_loss_0,
            "rotation": custom_loss_1,
        }
        loss_weights = {
            "translation": 42.0,
            "rotation": (0.1, 1.0),
        }

        fm = FlowMatching(
            reverse_fn=reverse_fn,
            sigma_min=0.0,
            inference_steps=10,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
        )

        # overwrite random methods
        def new_generate_target_tensor(x0, x1):
            return x1 + 1.0

        def new_generate_xt_tensor(x0, x1, t):
            return x1

        fm._generate_xt_tensor = new_generate_xt_tensor
        fm._generate_target_tensor = new_generate_target_tensor

        loss, detail_losses = fm.loss(x)
        self.assertEqual(type(detail_losses), dict)

        self.assertAlmostEqual(
            loss.item(),
            4.0 * 42.0 + 0.1 * 8.0 + 1.0 * 8.0,
            places=4,
        )

    def test_flow_matching(self):
        xs = [
            # single tensor
            torch.rand(size=(4, 32, 32)),
            # complex structure
            {
                "translation": torch.rand(size=(4, 32, 32)),
                "rotation": (
                    torch.rand(size=(4, 16, 16)),
                    torch.rand(size=(4, 8, 8)),
                ),
            },
        ]

        devices = ["cpu"]

        if torch.cuda.is_available():
            devices.append("cuda")
        else:
            logger.warning("cuda is not available, skipping cuda testing ...")

        reverse_fn = lambda x, _: x

        # crazy custom losses
        custom_loss_0 = Mock(return_value=torch.tensor(1.0))
        custom_loss_1 = Mock(return_value=torch.tensor(2.0))

        fms = [
            FlowMatching(reverse_fn=reverse_fn, sigma_min=0.1, inference_steps=10),
            # test uniform sampler
            FlowMatching(
                reverse_fn=reverse_fn,
                sigma_min=0.1,
                inference_steps=10,
                training_time_sampler_fn=uniform_sampler,
            ),
            # test structure loss broadcasting
            FlowMatching(
                reverse_fn=reverse_fn,
                sigma_min=0.1,
                inference_steps=10,
                loss_fn={
                    "translation": custom_loss_0,
                    "rotation": custom_loss_1,
                },
            ),
            FlowMatching(
                reverse_fn=reverse_fn,
                sigma_min=0.1,
                inference_steps=10,
                loss_fn={
                    "translation": custom_loss_0,
                    "rotation": custom_loss_1,
                },
                loss_weights={
                    "translation": 2.0,
                    "rotation": (0.1, 1.0),
                },
            ),
            FlowMatching(
                reverse_fn=reverse_fn,
                sigma_min=0.1,
                inference_steps=10,
                loss_fn={
                    "translation": custom_loss_0,
                    "rotation": custom_loss_1,
                },
                solver_method="midpoint",
            ),
            FlowMatching(
                reverse_fn=reverse_fn,
                sigma_min=0.1,
                inference_steps=10,
                loss_fn={
                    "translation": custom_loss_0,
                    "rotation": custom_loss_1,
                },
                solver_method="rk4",
            ),
        ]

        for fm in fms:
            for x in xs:
                for device in devices:
                    self._test_flow_matching(fm, x, device=device)

        self.assertEqual(custom_loss_0.call_count, (2 + 2 + 2 + 2) * len(devices))
        self.assertEqual(custom_loss_1.call_count, (3 + 3 + 4 + 3) * len(devices))


if __name__ == "__main__":
    run_unittest(UnitTests)
