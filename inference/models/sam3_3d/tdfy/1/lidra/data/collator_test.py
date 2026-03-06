import unittest
import torch

from lidra.data.collator import auto_collate, auto_uncollate
from lidra.test.util import run_unittest, OverwriteTensorEquality


class UnitTests(unittest.TestCase):
    BATCHED_STRINGS = ["a", "b", "c"]
    BATCHED_INTS = [0, 1, 2]
    BATCHED_TENSOR = torch.rand((3, 16, 4))

    def _test_batch(self, collate_fn, uncollate_fn, batch, expected_collate_output):
        with OverwriteTensorEquality(torch, check_shape=True):
            collated = collate_fn(batch)
            uncollated = uncollate_fn(collated)

            self.assertEqual(collated, expected_collate_output)
            self.assertEqual(batch, uncollated)

    def _gen_batch(self, type):
        if type == "dict":

            def make(*args):
                return {f"key_{i}": v for i, v in enumerate(args)}

        elif type == "tuple":

            def make(*args):
                return tuple(args)

        elif type == "list":

            def make(*args):
                return list(args)

        def gen(i):
            return make(
                UnitTests.BATCHED_STRINGS[i],
                UnitTests.BATCHED_INTS[i],
                UnitTests.BATCHED_TENSOR[i],
            )

        expected = make(
            UnitTests.BATCHED_STRINGS,
            UnitTests.BATCHED_INTS,
            UnitTests.BATCHED_TENSOR,
        )
        batch = [gen(i) for i in range(len(UnitTests.BATCHED_INTS))]

        return batch, expected

    def test_auto_collate(self):

        collate_fn = auto_collate()
        uncollate_fn = auto_uncollate()
        self._test_batch(collate_fn, uncollate_fn, *self._gen_batch(type="dict"))
        self._test_batch(collate_fn, uncollate_fn, *self._gen_batch(type="tuple"))
        self._test_batch(collate_fn, uncollate_fn, *self._gen_batch(type="list"))


if __name__ == "__main__":
    run_unittest(UnitTests)
