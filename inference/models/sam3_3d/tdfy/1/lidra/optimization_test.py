import unittest
import torch
from tqdm import tqdm
from functools import partial

from lidra.optimization import optimize
from lidra.test.util import run_unittest, run_only_if_cuda_is_available
from lidra.data.utils import build_batch_extractor


class UnitTests(unittest.TestCase):
    @run_only_if_cuda_is_available(default_device="cuda")
    def test_optimize(self):
        def dataloader(m=100):
            for idx in range(m):
                yield (
                    torch.rand(4, 32, device="cuda"),
                    torch.rand(4, 16, device="cuda"),
                    idx,
                )

        input_extractor = build_batch_extractor(0)
        target_extractor = build_batch_extractor(1)
        model = torch.nn.Linear(32, 16)

        iterable = optimize(
            dataloader=dataloader(),
            input_extractor_fn=input_extractor,
            target_extractor_fn=target_extractor,
            model=model,
            optimizer_fn=partial(torch.optim.SGD, lr=1e-4),
            loss_fn=torch.nn.MSELoss(reduction="mean"),
            max_iteration=10,
            device="cuda",
        )
        for idx, state in tqdm(enumerate(iterable)):
            self.assertEqual(idx, state.idx)
            self.assertTrue(idx < 10)
            self.assertIs(model, state.model)
            self.assertSequenceEqual((4, 32), state.x[0][0].shape)
            self.assertSequenceEqual((4, 16), state.y.shape)
            self.assertSequenceEqual((4, 16), state.target[0][0].shape)


if __name__ == "__main__":
    run_unittest(UnitTests)
