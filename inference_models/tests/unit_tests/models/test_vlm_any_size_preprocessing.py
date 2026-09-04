from typing import List
from unittest.mock import MagicMock

import numpy as np
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    AnySizePadding,
    ColorMode,
    ImagePreProcessing,
    InferenceConfig,
    NetworkInputDefinition,
    ResizeMode,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner
from inference_models.models.florence2.florence2_hf import Florence2HF
from inference_models.models.gemma4.gemma4_hf import Gemma4HF
from inference_models.models.paligemma.paligemma_hf import PaliGemmaHF
from inference_models.models.qwen25vl.qwen25vl_hf import Qwen25VLHF
from inference_models.models.smolvlm.smolvlm_hf import SmolVLMHF


def _any_size_inference_config() -> InferenceConfig:
    inference_config = InferenceConfig(
        image_pre_processing=ImagePreProcessing(),
        network_input=NetworkInputDefinition(
            training_input_size=None,
            dynamic_spatial_size_supported=True,
            dynamic_spatial_size_mode=AnySizePadding(type="any-size"),
            color_mode=ColorMode.RGB,
            resize_mode=ResizeMode.STRETCH_TO,
            input_channels=3,
        ),
    )

    return inference_config


def _heterogeneous_images() -> List[np.ndarray]:
    images = [
        np.zeros((20, 30, 3), dtype=np.uint8),
        np.zeros((40, 50, 3), dtype=np.uint8),
    ]

    return images


def _assert_tensor_image_shapes(images: List[torch.Tensor]) -> None:
    assert [tuple(image.shape) for image in images] == [
        (3, 20, 30),
        (3, 40, 50),
    ]


def test_smolvlm_preserves_heterogeneous_images_for_the_processor() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = ["templated"]
    model = SmolVLMHF(
        model=MagicMock(),
        processor=processor,
        inference_config=_any_size_inference_config(),
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
    )

    model.pre_process_generation(
        images=_heterogeneous_images(),
        prompt="Describe the images.",
        input_color_format="rgb",
    )

    _assert_tensor_image_shapes(processor.call_args.kwargs["images"])


def test_paligemma_preserves_heterogeneous_images_for_the_processor() -> None:
    processor = MagicMock()
    model = PaliGemmaHF(
        model=MagicMock(),
        processor=processor,
        inference_config=_any_size_inference_config(),
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
    )

    model.pre_process_generation(
        images=_heterogeneous_images(),
        prompt="Describe the images.",
        input_color_format="rgb",
    )

    _assert_tensor_image_shapes(processor.call_args.kwargs["images"])


def test_qwen25vl_preserves_heterogeneous_images_for_the_processor() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "templated"
    processor.return_value = {"pixel_values": torch.zeros((2, 3, 8, 8))}
    model = Qwen25VLHF(
        model=MagicMock(),
        processor=processor,
        inference_config=_any_size_inference_config(),
        device=torch.device("cpu"),
    )

    model.pre_process_generation(
        images=_heterogeneous_images(),
        prompt="Describe the images.",
        input_color_format="rgb",
    )

    _assert_tensor_image_shapes(processor.call_args.kwargs["images"])


def test_gemma4_preserves_heterogeneous_images_for_the_processor() -> None:
    processor = MagicMock()
    model = Gemma4HF(
        model=MagicMock(),
        processor=processor,
        inference_config=_any_size_inference_config(),
        device=torch.device("cpu"),
    )

    model.pre_process_generation(
        images=_heterogeneous_images(),
        prompt="Describe the images.",
        input_color_format="rgb",
    )

    conversation = processor.apply_chat_template.call_args.args[0]
    image_content = conversation[1]["content"][:-1]
    assert [entry["image"].size for entry in image_content] == [(30, 20), (50, 40)]


def test_florence2_preserves_heterogeneous_images_for_the_processor() -> None:
    processor = MagicMock()
    model = Florence2HF(
        model=MagicMock(),
        processor=processor,
        inference_config=_any_size_inference_config(),
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
    )

    _, image_dimensions, metadata = model.pre_process_generation(
        images=_heterogeneous_images(),
        prompt="Describe the images.",
        input_color_format="rgb",
    )

    _assert_tensor_image_shapes(processor.call_args.kwargs["images"])
    assert image_dimensions == [
        ImageDimensions(height=20, width=30),
        ImageDimensions(height=40, width=50),
    ]
    assert metadata is not None


def test_cosmos3_preserves_heterogeneous_images_for_the_processor() -> None:
    model_weights = MagicMock()
    model_weights.parameters.return_value = iter([torch.tensor(0.0)])
    processor = MagicMock()
    processor.apply_chat_template.return_value = "templated"
    processor.return_value = {"input_ids": torch.tensor([[1, 2]])}
    model = Cosmos3EdgeReasoner(
        model=model_weights,
        processor=processor,
        inference_config=_any_size_inference_config(),
        device=torch.device("cpu"),
    )

    model.pre_process_generation(
        images=_heterogeneous_images(),
        prompt="Describe the images.",
        input_color_format="rgb",
    )

    _assert_tensor_image_shapes(processor.call_args.kwargs["images"])
