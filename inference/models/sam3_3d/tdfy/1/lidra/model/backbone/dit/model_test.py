import unittest
import torch

from lidra.model.backbone.dit.model import (
    DiT,
    ImageToTokens,
    LabelEmbedder,
    Dino,
    unconditional,
    create_dit_backbone as cdb,
    VALID_CONDITION_TYPES,
    VALID_INPUT_TYPES,
)
from lidra.test.util import run_unittest, run_only_if_cuda_is_available


class UnitTests(unittest.TestCase):
    IMAGE_CHANNELS = 3
    INPUT_N_TOKENS = 32
    CONDITION_N_TOKENS = 16

    def _check_output(self, input, output):
        self.assertEqual(output.shape, input.shape)

    def _create_condition(self, model, batch_size):
        if isinstance(model.condition_embedder, LabelEmbedder):
            condition = torch.randint(
                low=0,
                high=model.condition_embedder.num_classes - 1,
                size=(batch_size,),
            )
        elif isinstance(model.condition_embedder, ImageToTokens) or isinstance(
            model.condition_embedder, Dino
        ):
            condition = torch.rand(
                (
                    batch_size,
                    model.condition_embedder.input_channels,
                    model.condition_embedder.input_size,
                    model.condition_embedder.input_size,
                )
            )
        elif model.condition_embedder is unconditional:
            condition = None
        else:
            condition = torch.rand(
                (
                    batch_size,
                    UnitTests.INPUT_N_TOKENS,
                    model.hidden_size,
                )
            )
        return condition

    def _create_input(self, model, batch_size):
        if isinstance(model.input_embedder, ImageToTokens):
            input = torch.rand(
                (
                    batch_size,
                    model.input_embedder.input_channels,
                    model.input_embedder.input_size,
                    model.input_embedder.input_size,
                )
            )
        elif isinstance(model.input_embedder, torch.nn.Linear):
            input = torch.rand(
                (
                    batch_size,
                    UnitTests.INPUT_N_TOKENS,
                    model.input_embedder.in_features,
                )
            )
        elif isinstance(model.input_embedder, torch.nn.Sequential):
            input = torch.rand(
                (
                    batch_size,
                    UnitTests.INPUT_N_TOKENS,
                    model.input_embedder[0].in_features,
                )
            )
        else:
            input = torch.rand(
                (
                    batch_size,
                    UnitTests.INPUT_N_TOKENS,
                    model.hidden_size,
                )
            )
        return input

    def _test_dit(
        self,
        model: DiT,
        batch_size=2,
        end_time=100,
    ):
        input = self._create_input(model, batch_size)
        time = torch.randint(low=0, high=end_time, size=(batch_size,))
        condition = self._create_condition(model, batch_size)

        if model.learn_sigma:
            output0, output1 = model(input, time, condition)
            self._check_output(input, output0)
            self._check_output(input, output1)
        else:
            new_image = model(input, time, condition)
            self._check_output(input, new_image)

    @staticmethod
    def merge_args(kwargs: dict, **ow_kwargs: dict):
        return {**kwargs, **ow_kwargs}

    @run_only_if_cuda_is_available(default_device="cuda")
    def test_dit(self):
        for input_type in VALID_INPUT_TYPES:
            for condition_type in VALID_CONDITION_TYPES:
                self._test_one_dit(
                    {
                        "input_type": input_type,
                        "condition_type": condition_type,
                    },
                )

    def _test_one_dit(self, base_args):
        default_args = {
            "input_size": 32,
            "cond_image_size": None,
            "patch_size": 2,
            "input_channels": UnitTests.IMAGE_CHANNELS,
            "hidden_size": 128,
            "depth": 4,
            "num_heads": 2,
            "num_classes": 12,
            "learn_sigma": True,
            "n_tokens": 32,
        }

        default_args = UnitTests.merge_args(default_args, **base_args)

        self._test_dit(cdb(**default_args))
        self._test_dit(cdb(**default_args), batch_size=4)
        self._test_dit(cdb(**default_args), end_time=50)

        self._test_dit(cdb(**UnitTests.merge_args(default_args, input_size=64)))
        self._test_dit(cdb(**UnitTests.merge_args(default_args, cond_image_size=64)))
        self._test_dit(cdb(**UnitTests.merge_args(default_args, patch_size=4)))
        self._test_dit(cdb(**UnitTests.merge_args(default_args, input_channels=4)))
        self._test_dit(cdb(**UnitTests.merge_args(default_args, hidden_size=64)))
        self._test_dit(cdb(**UnitTests.merge_args(default_args, depth=2)))
        self._test_dit(cdb(**UnitTests.merge_args(default_args, num_heads=1)))
        self._test_dit(cdb(**UnitTests.merge_args(default_args, num_classes=4)))
        self._test_dit(cdb(**UnitTests.merge_args(default_args, learn_sigma=False)))


if __name__ == "__main__":
    run_unittest(UnitTests)
