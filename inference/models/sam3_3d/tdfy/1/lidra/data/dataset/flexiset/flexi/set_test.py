import unittest

from lidra.test.util import run_unittest
from lidra.data.dataset.flexiset.flexi.set import Set as FlexiSet
from lidra.data.dataset.flexiset.flexi.loader import Loader as FlexiLoader
from lidra.data.dataset.flexiset.loaders.ops.lambda_ import Lambda


class UnitTests(unittest.TestCase):
    def test_flexi_set_simple_case(self):
        loaders = [
            FlexiLoader(
                "simple_add",
                Lambda(lambda x, y=0: x + y),
            ),
        ]

        fs = FlexiSet(
            inputs=("x", "y"),
            loaders=loaders,
            outputs=["simple_add"],
        )

        # should crash if no inputs are provided
        with self.assertRaises(ValueError):
            item = fs()  # missing inputs

        # test simple call
        item = fs(x=5, y=10)
        self.assertTrue("simple_add" in item)
        self.assertEqual(item["simple_add"], 15)

        # don't allow the use of optional inputs for now
        with self.assertRaises(ValueError):
            item = fs(x=5)

        # ... only if we respecify the inputs
        fs.inputs = "x"
        item = fs(x=5)
        self.assertTrue("simple_add" in item)
        self.assertEqual(item["simple_add"], 5)

        # test requesting inputs
        fs.inputs = ("x", "y")
        fs.outputs = ("x", "y")
        item = fs(x=5, y=10)
        self.assertEqual(item["x"], 5)
        self.assertEqual(item["y"], 10)

    def test_flexi_set_multi_node_case(self):
        def simple_add(x, y=0):
            simple_add.called += 1
            return x + y

        def simple_mul(x, y=0):
            simple_mul.called += 1
            return x * y

        def simple_inc(z):
            simple_inc.called += 1
            return z + 1

        def reset_called():
            simple_add.called = 0
            simple_mul.called = 0
            simple_inc.called = 0

        reset_called()

        loaders = [
            FlexiLoader(
                "simple_add",
                Lambda(simple_add),
            ),
            FlexiLoader(
                "simple_mul",
                Lambda(simple_mul),
            ),
            FlexiLoader(
                "inc_0",
                Lambda(simple_inc),
            ),
            FlexiLoader(
                "inc_1",
                Lambda(simple_inc),
                input_mapping={"z": "inc_0"},
            ),
        ]

        fs = FlexiSet(
            inputs=("x", "y", "z"),
            loaders=loaders,
            outputs=(),
        )

        # test with empty outputs
        item = fs(x=3.14, y=10, z=0)
        self.assertIsInstance(item, dict)
        self.assertTrue(len(item) == 0)
        self.assertEqual(simple_add.called, 0)
        self.assertEqual(simple_mul.called, 0)
        self.assertEqual(simple_inc.called, 0)

        # test minimal computation #1
        fs.outputs = "simple_mul"
        item = fs(x=3.14, y=10, z=0)
        self.assertTrue(len(item) == 1)
        self.assertTrue("simple_mul" in item)
        self.assertEqual(item["simple_mul"], 3.14 * 10)
        self.assertEqual(simple_add.called, 0)
        self.assertEqual(simple_mul.called, 1)
        self.assertEqual(simple_inc.called, 0)
        reset_called()

        # test minimal computation #2
        fs.outputs = "simple_add"
        item = fs(x=3.14, y=10, z=0)
        self.assertTrue(len(item) == 1)
        self.assertTrue("simple_add" in item)
        self.assertEqual(item["simple_add"], 13.14)
        self.assertEqual(simple_add.called, 1)
        self.assertEqual(simple_mul.called, 0)
        self.assertEqual(simple_inc.called, 0)
        reset_called()

        # test minimal computation #3
        fs.outputs = "inc_0"
        item = fs(x=3.14, y=10, z=0)
        self.assertTrue(len(item) == 1)
        self.assertTrue("inc_0" in item)
        self.assertEqual(item["inc_0"], 1)
        self.assertEqual(simple_add.called, 0)
        self.assertEqual(simple_mul.called, 0)
        self.assertEqual(simple_inc.called, 1)
        reset_called()

        # test minimal computation #4
        fs.outputs = "inc_1"
        item = fs(x=3.14, y=10, z=0)
        self.assertTrue(len(item) == 1)
        self.assertTrue("inc_1" in item)
        self.assertEqual(item["inc_1"], 2)
        self.assertEqual(simple_add.called, 0)
        self.assertEqual(simple_mul.called, 0)
        self.assertEqual(simple_inc.called, 2)
        reset_called()


if __name__ == "__main__":
    run_unittest(UnitTests)
