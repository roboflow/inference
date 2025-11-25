import unittest
import torch

from lidra.model.backbone.generator.ddpm.model import DDPM
from lidra.model.backbone.generator.ddpm.schedule import linear, Schedule
from lidra.test.util import run_unittest


class UnitTests(unittest.TestCase):
    # TODO(Pierre) : write better unit tests when generator's interface is finalized
    def test_ddpm(self):
        schedule = Schedule(
            schedule_fn=linear(),
            max_t=1000,
        )
        schedule.max_t = 10
        ddpm = DDPM(
            reverse_fn=lambda x, _: x,
            schedule=schedule,
        )

        loss = ddpm.loss(torch.rand(size=(4, 32, 32)))
        x = ddpm.generate((4, 32, 32), x_device="cpu")


if __name__ == "__main__":
    run_unittest(UnitTests)
