import unittest
import torch

from lidra.model.backbone.generator.ddim.model import DDIM
from lidra.model.backbone.generator.ddpm.schedule import (
    linear,
    Schedule,
    LinearSubSchedule,
)
from lidra.test.util import run_unittest


class UnitTests(unittest.TestCase):
    # TODO(Pierre) : write better unit tests when generator's interface is finalized
    def test_ddim(self):
        training_schedule = Schedule(
            schedule_fn=linear(),
            max_t=1000,
        )
        generative_schedule = LinearSubSchedule(training_schedule, max_t=10)

        ddim = DDIM(
            reverse_fn=lambda x, _: x,
            training_schedule=training_schedule,
            generative_schedule=generative_schedule,
            eta=1,
        )

        loss = ddim.loss(torch.rand(size=(4, 32, 32)))
        x = ddim.generate((4, 32, 32), x_device="cpu")


if __name__ == "__main__":
    run_unittest(UnitTests)
