import unittest
import torch

from lidra.test.util import run_unittest, temporary_file
from lidra.test.model import MockMLP
import lidra.mixhavior as mixhavior
from lidra.mixhavior.pytorch import PyTorchMixhaviorStateDictHandling


class BehaviorA(mixhavior.Behavior):
    def _setup(self, mixh):
        self.buffer = torch.zeros((16, 16))

    def _attach(self, mixh):
        # make persistent buffer and make sure it's not include in state_dict
        mixh.obj.register_buffer("adhoc_buffer", self.buffer, persistent=True)

    def _detach(self):
        delattr(self.obj, "adhoc_buffer")

    def inc(self):
        self.buffer += 1


class UnitTests(unittest.TestCase):
    def test_pytorch_mixhavior_state_dict_handling(self):
        # create mock model and equip it to handle mixhavior stuff in state dict
        model_0 = MockMLP()
        mixh_0 = mixhavior.get_mixhavior(model_0)
        bid_0 = mixh_0.equip(PyTorchMixhaviorStateDictHandling())
        # add mock behavior with internal data
        bid_1 = mixh_0.equip(BehaviorA())

        # update internal data using behavior
        mixh_0[bid_1].inc()

        # serialize model and load the data
        with temporary_file() as tmp_path:
            torch.save(model_0.state_dict(), tmp_path)
            data = torch.load(tmp_path, weights_only=False)

        # make sure buffer has been properly detached before being put in dict
        self.assertEqual("adhoc_buffer" in data, False)
        # make sure mixhavior data has been serialized
        assert "__mixhavior__" in model_0.state_dict()

        # create a copy of the model, equipped with state dict handling behavior
        model_1 = MockMLP()
        mixh_1 = mixhavior.get_mixhavior(model_1)
        mixh_1.equip(PyTorchMixhaviorStateDictHandling())

        # load the data
        model_1.load_state_dict(data, strict=True)

        # update internal data using behavior again (to check its not changing previous model's data)
        mixh_1[bid_1].inc()

        # previous model has not changed
        self.assertEqual(mixh_0[bid_1].buffer.sum(), mixh_0[bid_1].buffer.numel())
        self.assertEqual(model_0.adhoc_buffer.sum(), model_0.adhoc_buffer.numel())

        # new model call inc() twice
        self.assertEqual(mixh_1[bid_1].buffer.sum(), mixh_1[bid_1].buffer.numel() * 2)
        self.assertEqual(model_1.adhoc_buffer.sum(), model_1.adhoc_buffer.numel() * 2)

        # test without the "__mixhavior__" key in state dict
        model_2 = MockMLP()
        mixh_2 = mixhavior.get_mixhavior(model_2)
        mixh_2.equip(PyTorchMixhaviorStateDictHandling())
        del data["__mixhavior__"]
        model_2.load_state_dict(data, strict=True)

        self.assertFalse(hasattr(model_2, "adhoc_buffer"))
        self.assertEqual(len(mixh_2.behaviors), 1)


if __name__ == "__main__":
    run_unittest(UnitTests)
