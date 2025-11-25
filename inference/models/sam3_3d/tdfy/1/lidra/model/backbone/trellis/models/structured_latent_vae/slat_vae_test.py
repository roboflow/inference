import unittest
from omegaconf import OmegaConf
import torch
import os
from hydra.utils import instantiate

from lidra.test.util import run_unittest, run_only_if_cuda_is_available
from lidra.model.backbone.trellis.modules import sparse as sp


class UnitTests(unittest.TestCase):
    DEVICE = "cuda:0"

    @run_only_if_cuda_is_available(default_device="cuda")
    def test_load_pretrained_slat_decoder_gs(self):
        slat_decoder_gs_config_path = os.path.join(
            "./etc/lidra/model/backbone/trellis_vae/slat_decoder_gs.yaml"
        )  # config used for the training
        slat_decoder_gs_config = OmegaConf.load(slat_decoder_gs_config_path)
        slat_decoder_gs = instantiate(slat_decoder_gs_config)
        slat_decoder_gs = slat_decoder_gs.eval()
        slat_decoder_gs = slat_decoder_gs.to(self.DEVICE)

        n = 5000
        coords = torch.randint(0, 64, (n, 3))
        coords = torch.cat([torch.zeros(n, 1), coords], dim=-1)
        coords = coords.int().to(self.DEVICE)
        input_sp = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], 8).to(self.DEVICE),
            coords=coords,
        )
        slat_decoder_gs(input_sp)


if __name__ == "__main__":
    run_unittest(UnitTests)
