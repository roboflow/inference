import unittest
import torch

from lidra.model.backbone.unet.diffusion import UNet
from lidra.test.util import run_unittest


class UnitTests(unittest.TestCase):
    IMAGE_CHANNELS = 3

    def _test_diffusion_unet(
        self,
        model: UNet,
        batch_size=2,
        end_time=100,
        input_size=32,
    ):

        # create fake image and positions
        image = torch.rand(
            (
                batch_size,
                model.in_channels,
                input_size,
                input_size,
            )
        )
        time = torch.randint(low=0, high=end_time, size=(batch_size,))

        new_image = model(image, time)

        # check output
        self.assertEqual(new_image.ndim, 4)
        self.assertEqual(new_image.shape[0], batch_size)
        self.assertEqual(new_image.shape[1], model.out_channels)
        self.assertEqual(new_image.shape[2], input_size)
        self.assertEqual(new_image.shape[3], input_size)

    @staticmethod
    def merge_args(kwargs: dict, **ow_kwargs: dict):
        return {**kwargs, **ow_kwargs}

    def test_diffusion_unet(self):
        default_args = {
            "in_channels": UnitTests.IMAGE_CHANNELS,
            "hid_channels": 128,
            "out_channels": UnitTests.IMAGE_CHANNELS,
            "ch_multipliers": (1, 2, 3),
            "num_res_blocks": 2,
            "apply_attn": (False, True, False),
        }
        self._test_diffusion_unet(UNet(**default_args))
        self._test_diffusion_unet(UNet(**default_args), batch_size=4)
        self._test_diffusion_unet(UNet(**default_args), end_time=50)
        self._test_diffusion_unet(UNet(**default_args), input_size=64)

        self._test_diffusion_unet(UNet(**UnitTests.merge_args(default_args, in_channels=4)))  # fmt: skip
        self._test_diffusion_unet(UNet(**UnitTests.merge_args(default_args, hid_channels=64)))  # fmt: skip
        self._test_diffusion_unet(UNet(**UnitTests.merge_args(default_args, out_channels=6)))  # fmt: skip
        self._test_diffusion_unet(UNet(**UnitTests.merge_args(default_args, ch_multipliers=(1, 2, 1))))  # fmt: skip
        self._test_diffusion_unet(UNet(**UnitTests.merge_args(default_args, num_res_blocks=3)))  # fmt: skip
        self._test_diffusion_unet(UNet(**UnitTests.merge_args(default_args, apply_attn=(True, True, True))))  # fmt: skip


if __name__ == "__main__":
    run_unittest(UnitTests)
