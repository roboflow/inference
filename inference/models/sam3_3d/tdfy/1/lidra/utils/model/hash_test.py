import copy
import unittest
import torch

from lidra.utils.model.hash import hash_module, diff_hashed_model
from lidra.test.model import MockMLP
from lidra.test.util import run_unittest


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.model_0 = MockMLP()
        self.model_1 = MockMLP()

    def test_hash(self):
        model_2 = copy.deepcopy(self.model_0)

        p_hash_0 = hash_module(self.model_0)
        p_hash_1 = hash_module(self.model_1)
        p_hash_2 = hash_module(model_2)

        # simple equality test
        self.assertEqual(p_hash_0, p_hash_2)
        self.assertNotEqual(p_hash_0, p_hash_1)  # different init

        # test adding new parameter
        model_2._new_module = torch.nn.Linear(2, 2)
        p_hash_2 = hash_module(model_2)
        self.assertNotEqual(p_hash_0, p_hash_2)

        delta = diff_hashed_model(p_hash_0, p_hash_2)
        self.assertEqual(delta["buffers"], {})
        for key in delta["parameters"]:
            self.assertTrue(key.startswith("_new_module"))

        # test adding new buffer
        model_2.register_buffer("_new_buffer", torch.zeros(1))
        p_hash_2 = hash_module(model_2)
        self.assertNotEqual(p_hash_0, p_hash_2)

        delta = diff_hashed_model(p_hash_0, p_hash_2)
        for key in delta["parameters"]:
            self.assertTrue(key.startswith("_new_module"))
        for key in delta["buffers"]:
            self.assertTrue(key.startswith("_new_buffer"))


if __name__ == "__main__":
    run_unittest(UnitTests)
