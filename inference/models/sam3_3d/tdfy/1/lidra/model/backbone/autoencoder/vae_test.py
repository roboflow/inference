import unittest
import torch

from lidra.model.backbone.autoencoder.vae import KLAutoEncoder
from lidra.test.util import run_unittest


class UnitTests(unittest.TestCase):
    POINT_CLOUD_SIZE = 1024

    def _model_forward(
        self,
        model: KLAutoEncoder,
        batch_size=2,
        query_size=550,
    ):

        # create fake point cloud and positions
        point_cloud = torch.rand(
            (
                batch_size,
                UnitTests.POINT_CLOUD_SIZE,
                3,
            )
        )
        xyz = torch.rand((batch_size, query_size, 3))

        result = model(point_cloud, xyz)
        logits, kl = result["logits"], result["kl"]

        # check logit output
        self.assertEqual(logits.ndim, 2)
        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], query_size)

        # check kl output
        self.assertEqual(kl.ndim, 1)
        self.assertEqual(kl.shape[0], batch_size)

    def _test_vae_dimensions(self, model):
        # default parameters
        self._model_forward(model)

        # vary one dimension
        self._model_forward(model, batch_size=3)
        self._model_forward(model, query_size=256)

    def test_vae(self):
        self._test_vae_dimensions(KLAutoEncoder(n_blocks=12))
        self._test_vae_dimensions(KLAutoEncoder(embed_dim=128, queries_dim=128))
        self._test_vae_dimensions(KLAutoEncoder(num_latents=128))
        self._test_vae_dimensions(KLAutoEncoder(latent_dim=32))
        self._test_vae_dimensions(KLAutoEncoder(num_heads=32))


if __name__ == "__main__":
    run_unittest(UnitTests)
