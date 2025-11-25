import unittest
import torch

from lidra.model.backbone.mcc.model import MCC
from lidra.test.util import run_unittest


class UnitTests(unittest.TestCase):
    def _model_forward(
        self,
        model: MCC,
        batch_size=2,
        image_size=512,
        query_size=550,
    ):

        # create fake image and positions
        image = torch.rand((batch_size, MCC.IMAGE_CHANNELS, image_size, image_size))
        xyz = torch.rand((batch_size, query_size, 3))

        if model.color_prediction:
            occ, color = model(image, xyz)

            # check color output
            self.assertEqual(color.ndim, 3)
            self.assertEqual(color.shape[0], batch_size)
            self.assertEqual(color.shape[1], query_size)
            self.assertEqual(color.shape[2], 256 * MCC.IMAGE_CHANNELS)
        else:
            occ = model(image, xyz)

        # check occ output
        self.assertEqual(occ.ndim, 2)
        self.assertEqual(occ.shape[0], batch_size)
        self.assertEqual(occ.shape[1], query_size)

    def _test_mcc_dimensions(self, model):
        # default parameters
        self._model_forward(model)

        # vary one dimension
        self._model_forward(model, batch_size=3)
        self._model_forward(model, image_size=256)
        self._model_forward(model, query_size=1024)

    def test_mcc(self):
        self._test_mcc_dimensions(MCC(color_prediction=False))
        self._test_mcc_dimensions(MCC(color_prediction=True))


if __name__ == "__main__":
    run_unittest(UnitTests)
