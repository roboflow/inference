import unittest
import torch
from copy import deepcopy
from loguru import logger

from lidra.callback.ema import EMACache
from lidra.test.util import run_unittest, OverwriteTensorEquality, temporary_file
from lidra.test.model import MockMLP
import lidra.mixhavior as mixhavior
from lidra.mixhavior.pytorch import PyTorchMixhaviorStateDictHandling


class UnitTests(unittest.TestCase):
    @staticmethod
    def _new_model():
        model = MockMLP()

        if torch.cuda.is_available():
            model = model.cuda()

        return model

    def _model_equal(self, model_0, model_1):
        with OverwriteTensorEquality(
            torch,
            check_shape=True,
            check_device=True,
            check_dtype=True,
        ):
            p0 = tuple(p.data for p in model_0.parameters())
            p1 = tuple(p.data for p in model_1.parameters())
            self.assertEqual(p0, p1)

    def _model_not_equal(self, model_0, model_1):
        with OverwriteTensorEquality(
            torch,
            check_shape=True,
            check_device=True,
            check_dtype=True,
        ):
            self.assertNotEqual(model_0.state_dict(), model_1.state_dict())

    def _is_decayed(self, decay, n_times, model=None):
        model = self.model if model is None else model
        for param, ori_param in zip(
            model.parameters(),
            self.original_model.parameters(),
        ):
            decayed_param = (decay**n_times) * ori_param.data + (1.0 - decay**n_times)
            self.assertTrue(
                torch.allclose(param.data, decayed_param, atol=1e-6),
                f"tensors aren't close : {param.data} != {decayed_param} (max error = {(param.data - decayed_param).abs().max()})",
            )

    def setUp(self):
        # create mock model
        self.model = UnitTests._new_model()

        # keep a copy of the original model (without mixhavior)
        self.original_model = deepcopy(self.model)

        # add ema caches
        mixh = mixhavior.get_mixhavior(self.model)
        mixh.equip(PyTorchMixhaviorStateDictHandling())
        self.bids = [
            mixh.equip(EMACache(decay=0.9, device="cpu"), "ema_0.9"),
            mixh.equip(EMACache(decay=0.8, device="cpu"), "ema_0.8"),
            mixh.equip(EMACache(decay=0.7, device=None), "ema_0.7"),
        ]

        if torch.cuda.is_available():
            cuda_behavior = mixh.equip(EMACache(decay=0.6, device="cuda:0"), "ema_0.6")
            self.bids.append(cuda_behavior)
        else:
            logger.warning("could not test EMA on cuda device")

        # set model to ones
        for param in self.model.parameters():
            torch.fill_(param.data, 1.0)
        self.all_ones_model = deepcopy(self.model)

        # current model is all set to 0
        self._model_equal(self.model, self.all_ones_model)

    def test_ema_swap(self):
        # test swap behavior (train)
        mixh = mixhavior.get_mixhavior(self.model)
        ema_behavior: EMACache = mixh[self.bids[0]]

        self.model.train()
        ema_behavior.swap(ensure_in_state="ema")
        self._model_equal(self.model, self.original_model)
        ema_behavior.swap(ensure_in_state="model")
        self._model_equal(self.model, self.all_ones_model)

        self.model.eval()
        ema_behavior.swap(ensure_in_state="ema")
        self._model_equal(self.model, self.original_model)
        ema_behavior.swap(ensure_in_state="model")
        self._model_equal(self.model, self.all_ones_model)

    def test_ema_state(self):
        mixh = mixhavior.get_mixhavior(self.model)
        ema_behavior: EMACache = mixh[self.bids[0]]

        ema_behavior.ensure_in_state("ema")  # shouldn't change
        self._model_equal(self.model, self.all_ones_model)

        ema_behavior.ensure_in_state("model")  # should swap
        self._model_equal(self.model, self.original_model)
        ema_behavior.ensure_in_state("model")  # shouldn't change
        self._model_equal(self.model, self.original_model)

        ema_behavior.ensure_in_state("ema")  # revert back to original
        self._model_equal(self.model, self.all_ones_model)

        ema_behavior.ensure_in_state("model")  # revert back to original
        ema_behavior.remove()  # removing ema cache should revert to ema state
        self._model_equal(self.model, self.all_ones_model)

    def _test_ema_update(self, ema_behavior: EMACache):
        with ema_behavior.ema_loaded():
            self._is_decayed(ema_behavior.decay, n_times=0)

        # first update
        ema_behavior.ema_update()
        # model shouldn't change after update
        self._model_equal(self.model, self.all_ones_model)
        with ema_behavior.ema_loaded():
            self._is_decayed(ema_behavior.decay, n_times=1)

        # second update
        ema_behavior.ema_update()
        self._model_equal(self.model, self.all_ones_model)
        with ema_behavior.ema_loaded():
            self._is_decayed(ema_behavior.decay, n_times=2)

    def test_ema_update(self):
        mixh = mixhavior.get_mixhavior(self.model)
        ema_behavior: EMACache = mixh[self.bids[0]]
        self._test_ema_update(ema_behavior)

    def test_multi_ema(self):
        mixh = mixhavior.get_mixhavior(self.model)
        # reversed to test different flow from test_ema_update
        for bid in reversed(self.bids):
            ema_behavior: EMACache = mixh[bid]
            self._test_ema_update(ema_behavior)

    def test_serialization(self):
        self.test_multi_ema()

        with temporary_file() as tmp_path:
            torch.save(self.model.state_dict(), tmp_path)
            data = torch.load(tmp_path, weights_only=False)

        # check all saved tensors are on "cpu"
        for ema_cache in data["__mixhavior__"].values():
            for k, p in ema_cache.parameters():
                self.assertEqual(p.device.type, "cpu")

        # create new model and load it with emas
        model = UnitTests._new_model()
        mixh = mixhavior.get_mixhavior(model)
        mixh.equip(PyTorchMixhaviorStateDictHandling())
        model.load_state_dict(data, strict=True)

        # model hasn't changed
        self._model_equal(model, self.all_ones_model)

        # the decay persisted
        for bid in self.bids:
            ema_behavior: EMACache = mixh[bid]
            with ema_behavior.ema_loaded():
                self._is_decayed(ema_behavior.decay, n_times=2, model=model)


if __name__ == "__main__":
    run_unittest(UnitTests)
