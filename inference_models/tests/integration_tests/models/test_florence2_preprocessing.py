import numpy as np
import pytest
import torch

from inference_models.models.florence2.florence2_hf import Florence2HF


@pytest.fixture(scope="module")
def florence2_model(florence2_base_ft_path: str) -> Florence2HF:
    return Florence2HF.from_pretrained(florence2_base_ft_path)


def get_preprocessed_outputs(
    florence2_model: Florence2HF,
    dog_image_numpy: np.ndarray,
    dog_image_torch: torch.Tensor,
):
    prompt = "<OD>"
    # Process single numpy image (BGR)
    numpy_output, _, _ = florence2_model.pre_process_generation(
        images=dog_image_numpy, prompt=prompt
    )

    # Process single torch tensor (RGB)
    tensor_output, _, _ = florence2_model.pre_process_generation(
        images=dog_image_torch, prompt=prompt
    )

    # Process list of numpy images
    list_numpy_output, _, _ = florence2_model.pre_process_generation(
        images=[dog_image_numpy, dog_image_numpy], prompt=prompt
    )

    # Process list of torch tensors
    list_tensor_output, _, _ = florence2_model.pre_process_generation(
        images=[dog_image_torch, dog_image_torch], prompt=prompt
    )

    # Process batched tensor
    batched_tensor = torch.stack([dog_image_torch, dog_image_torch])
    batched_tensor_output, _, _ = florence2_model.pre_process_generation(
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
    florence2_model: Florence2HF,
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
    ) = get_preprocessed_outputs(florence2_model, dog_image_numpy, dog_image_torch)

    # THEN
    # Check shapes for single image inputs
    assert "pixel_values" in numpy_output and numpy_output["pixel_values"].shape[0] == 1
    assert (
        "pixel_values" in tensor_output and tensor_output["pixel_values"].shape[0] == 1
    )

    # Check shapes for multi-image inputs
    assert (
        "pixel_values" in list_numpy_output
        and list_numpy_output["pixel_values"].shape[0] == 2
    )
    assert (
        "pixel_values" in list_tensor_output
        and list_tensor_output["pixel_values"].shape[0] == 2
    )
    assert (
        "pixel_values" in batched_tensor_output
        and batched_tensor_output["pixel_values"].shape[0] == 2
    )


@pytest.mark.slow
@pytest.mark.hf_vlm_models
def test_internal_consistency_of_preprocessed_inputs(
    florence2_model: Florence2HF,
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
    ) = get_preprocessed_outputs(florence2_model, dog_image_numpy, dog_image_torch)

    # THEN
    # Compare single numpy (BGR) and single tensor (RGB)
    assert torch.allclose(
        numpy_output["pixel_values"], tensor_output["pixel_values"], atol=1e-2
    )
    assert torch.allclose(
        numpy_output["input_ids"], tensor_output["input_ids"], atol=1e-2
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
