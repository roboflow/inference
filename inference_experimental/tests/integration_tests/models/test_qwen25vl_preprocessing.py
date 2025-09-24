import numpy as np
import pytest
import torch
from inference_exp.models.qwen25vl.qwen25vl_hf import Qwen25VLHF


@pytest.fixture(scope="module")
def qwen_model(qwen25vl_3b_path: str) -> Qwen25VLHF:
    return Qwen25VLHF.from_pretrained(qwen25vl_3b_path)


def get_preprocessed_outputs(
    qwen_model: Qwen25VLHF,
    dog_image_numpy: np.ndarray,
    dog_image_torch: torch.Tensor,
):
    prompt = "What is in the image?"
    # Process single numpy image (BGR)
    numpy_output = qwen_model.pre_process_generation(
        images=dog_image_numpy, prompt=prompt
    )

    # Process single torch tensor (RGB)
    tensor_output = qwen_model.pre_process_generation(
        images=dog_image_torch, prompt=prompt
    )

    # Process list of numpy images
    list_numpy_output = qwen_model.pre_process_generation(
        images=[dog_image_numpy, dog_image_numpy], prompt=prompt
    )

    # Process list of torch tensors
    list_tensor_output = qwen_model.pre_process_generation(
        images=[dog_image_torch, dog_image_torch], prompt=prompt
    )

    # Process batched tensor
    batched_tensor = torch.stack([dog_image_torch, dog_image_torch])
    batched_tensor_output = qwen_model.pre_process_generation(
        images=batched_tensor, prompt=prompt
    )

    return (
        numpy_output,
        tensor_output,
        list_numpy_output,
        list_tensor_output,
        batched_tensor_output,
    )


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_preprocessed_output_shapes(
    qwen_model: Qwen25VLHF,
    dog_image_numpy: np.ndarray,
    dog_image_torch: torch.Tensor,
):
    # GIVEN
    (
        numpy_output,
        tensor_output,
        list_numpy_output,
        list_tensor_output,
        batched_tensor_output,
    ) = get_preprocessed_outputs(qwen_model, dog_image_numpy, dog_image_torch)

    # THEN
    # Check shapes for single image inputs
    assert (
        "image_grid_thw" in numpy_output
        and numpy_output["image_grid_thw"].shape[0] == 1
    )
    assert (
        "image_grid_thw" in tensor_output
        and tensor_output["image_grid_thw"].shape[0] == 1
    )

    # Check shapes for multi-image inputs
    assert (
        "image_grid_thw" in list_numpy_output
        and list_numpy_output["image_grid_thw"].shape[0] == 2
    )
    assert (
        "image_grid_thw" in list_tensor_output
        and list_tensor_output["image_grid_thw"].shape[0] == 2
    )
    assert (
        "image_grid_thw" in batched_tensor_output
        and batched_tensor_output["image_grid_thw"].shape[0] == 2
    )


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_internal_consistency_of_preprocessed_inputs(
    qwen_model: Qwen25VLHF,
    dog_image_numpy: np.ndarray,
    dog_image_torch: torch.Tensor,
):
    # GIVEN
    (
        numpy_output,
        tensor_output,
        list_numpy_output,
        list_tensor_output,
        batched_tensor_output,
    ) = get_preprocessed_outputs(qwen_model, dog_image_numpy, dog_image_torch)
    # The dog_image_numpy is BGR, dog_image_torch is RGB.
    # The processor should handle the conversion, but let's compare RGB numpy to RGB tensor
    prompt = "What is in the image?"
    rgb_dog_image_numpy = dog_image_numpy[:, :, ::-1]
    numpy_rgb_output = qwen_model.pre_process_generation(
        images=rgb_dog_image_numpy, prompt=prompt, input_color_format="rgb"
    )

    # THEN
    # Compare single numpy (RGB) and single tensor (RGB)
    assert torch.allclose(
        numpy_rgb_output["pixel_values"], tensor_output["pixel_values"], atol=1e-2
    )
    assert torch.allclose(
        numpy_rgb_output["input_ids"], tensor_output["input_ids"], atol=1e-2
    )

    # Compare list of tensors and batched tensor
    assert torch.allclose(
        list_tensor_output["pixel_values"],
        batched_tensor_output["pixel_values"],
        atol=1e-2,
    )
    assert torch.allclose(
        list_tensor_output["input_ids"],
        batched_tensor_output["input_ids"],
        atol=1e-2,
    )
