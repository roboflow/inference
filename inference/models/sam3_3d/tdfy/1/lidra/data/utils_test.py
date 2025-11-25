import unittest

from lidra.data.utils import build_batch_extractor
from lidra.test.util import run_unittest


class UnitTests(unittest.TestCase):
    def test_batch_extractor(self):
        batch = (1, 2, {"a": 3, "b": {"c": 4}, "ab": 5})

        extract_fn = build_batch_extractor(mapping=None)
        self.assertEqual(extract_fn(batch), ((batch,), {}))

        extract_fn = build_batch_extractor(mapping=1)
        self.assertEqual(extract_fn(batch), ((batch[1],), {}))

        extract_fn = build_batch_extractor(mapping="ab")
        self.assertEqual(extract_fn(batch[2]), ((batch[2]["ab"],), {}))

        extract_fn = build_batch_extractor(mapping=[1, (2, "b"), None])
        self.assertSequenceEqual(
            extract_fn(batch), ((batch[1], batch[2]["b"], batch), {})
        )

        extract_fn = build_batch_extractor(mapping={"A": 0, "B": (2, "a")})
        self.assertSequenceEqual(
            extract_fn(batch), ((), {"A": batch[0], "B": batch[2]["a"]})
        )

        extract_fn = build_batch_extractor(
            mapping=(
                [1, (2, "b"), None],
                {"A": 0, "B": (2, "a")},
            )
        )
        self.assertSequenceEqual(
            extract_fn(batch),
            (
                (batch[1], batch[2]["b"], batch),
                {"A": batch[0], "B": batch[2]["a"]},
            ),
        )


if __name__ == "__main__":
    run_unittest(UnitTests)
