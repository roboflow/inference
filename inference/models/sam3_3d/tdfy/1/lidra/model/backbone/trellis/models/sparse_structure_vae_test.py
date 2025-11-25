import unittest
import hydra
from lidra.utils.notebook.hydra import LidraConf
from omegaconf import OmegaConf
import torch
import os
from hydra.utils import instantiate

from lidra.test.util import (
    run_unittest,
    run_only_if_cuda_is_available,
    run_only_if_path_exists,
)


class UnitTests(unittest.TestCase):
    DEVICE = "cuda:0"

    @run_only_if_path_exists("/checkpoint/3dfy")
    @run_only_if_cuda_is_available(default_device="cuda")
    def test_load_pretrained_ss_decoder(self):
        ss_decoder_config_path = os.path.join(
            "model/backbone/trellis_vae/ss_decoder.yaml"
        )  # config used for the training
        ss_decoder_config = LidraConf.load_config(
            ss_decoder_config_path,
            overrides=["+cluster.path.weights=/checkpoint/3dfy/shared/weights/"],
        )
        ss_decoder = instantiate(
            ss_decoder_config,
        )
        ss_decoder = ss_decoder.eval()
        ss_decoder = ss_decoder.to(self.DEVICE)
        input_tensor = torch.rand(1, 8, 16, 16, 16).to(self.DEVICE)
        ss_decoder(input_tensor)

    @run_only_if_path_exists("/checkpoint/3dfy")
    @run_only_if_cuda_is_available(default_device="cuda")
    def test_load_pretrained_ss_decoder_non_exist_path(self):
        ss_decoder_config_path = os.path.join(
            "model/backbone/trellis_vae/ss_decoder.yaml"
        )  # config used for the training
        ss_decoder_config = LidraConf.load_config(
            ss_decoder_config_path,
            overrides=["+cluster.path.weights=/checkpoint/3dfy/shared/weights/"],
        )
        # non-exist path to test whether error is raised
        ss_decoder_config["pretrained_ckpt_path"] = (
            "/checkpoint/3dfy/shared/weights/tdfy/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8.safetensors"
        )
        with self.assertRaises(hydra.errors.InstantiationException) as e:
            ss_decoder = instantiate(ss_decoder_config)
        self.assertIsInstance(e.exception.__cause__, FileNotFoundError)

    @run_only_if_path_exists("/fsx-3dfy-v2/shared/weights/tdfy")
    def test_load_pretrained_ss_encoder(self):
        ss_encoder_config_path = os.path.join(
            "model/backbone/trellis_vae/ss_encoder.yaml"
        )
        ss_encoder_config = LidraConf.load_config(
            ss_encoder_config_path,
        )
        ss_encoder = instantiate(
            ss_encoder_config,
        )
        ss_encoder = ss_encoder.eval()
        ss_encoder = ss_encoder.to(self.DEVICE)
        input_tensor = torch.rand(1, 1, 64, 64, 64).to(self.DEVICE)
        ss_encoder(input_tensor)

    @run_only_if_path_exists("/fsx-3dfy-v2/shared/weights/tdfy")
    def test_load_pretrained_weiyao_encoder(self):
        ss_encoder_config_path = os.path.join(
            "model/backbone/trellis_vae/ss_encoder.yaml"
        )
        ss_encoder_config = LidraConf.load_config(
            ss_encoder_config_path,
        )
        ss_encoder_config["pretrained_ckpt_path"] = (
            "/fsx-3dfy-v2/shared/sparse-vae/trellis-ss-vae-trial1-lambda1e-3-8x8/weiyao_encoder.ckpt"
        )
        ss_encoder = instantiate(
            ss_encoder_config,
        )
        ss_encoder = ss_encoder.eval()
        ss_encoder = ss_encoder.to(self.DEVICE)
        input_tensor = torch.rand(1, 1, 64, 64, 64).to(self.DEVICE)
        ss_encoder(input_tensor)

    @run_only_if_path_exists("/fsx-3dfy-v2/shared/weights/tdfy")
    def test_load_pretrained_weiyao_decoder(self):
        ss_decoder_config_path = os.path.join(
            "model/backbone/trellis_vae/ss_decoder.yaml"
        )  # config used for the training
        ss_decoder_config = LidraConf.load_config(
            ss_decoder_config_path,
        )
        ss_decoder_config["pretrained_ckpt_path"] = (
            "/fsx-3dfy-v2/shared/sparse-vae/trellis-ss-vae-trial1-lambda1e-3-8x8/weiyao_decoder.ckpt"
        )
        ss_decoder = instantiate(
            ss_decoder_config,
        )
        ss_decoder = ss_decoder.eval()
        ss_decoder = ss_decoder.to(self.DEVICE)
        input_tensor = torch.rand(1, 8, 16, 16, 16).to(self.DEVICE)
        ss_decoder(input_tensor)


if __name__ == "__main__":
    run_unittest(UnitTests)
