import numpy as np
import pytest
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.errors import ModelInputError
from inference_exp.models.sam.sam_torch import SAMTorch


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_embeddings_numpy(sam_package: str, truck_image_numpy: np.ndarray) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)

    # when
    results = model.embed_images(truck_image_numpy)

    # then
    assert len(results) == 1
    assert 12400 <= results[0].embeddings.cpu().sum() <= 12500


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_embeddings_torch(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)

    # when
    results = model.embed_images(truck_image_torch)

    # then
    assert len(results) == 1
    assert 12400 <= results[0].embeddings.cpu().sum() <= 12500


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_without_prompting_numpy(
    sam_package: str, truck_image_numpy: np.ndarray
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)

    # when
    results = model.segment_images(truck_image_numpy)

    # then
    assert len(results) == 1
    assert np.allclose(
        results[0].scores.cpu().numpy(), np.array([0.8682, 0.7068, 0.4445]), atol=0.01
    )
    assert 724600 <= results[0].masks[0].cpu().sum() <= 725000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_without_prompting_batch_numpy(
    sam_package: str, truck_image_numpy: np.ndarray
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)

    # when
    results = model.segment_images([truck_image_numpy, truck_image_numpy])

    # then
    assert len(results) == 2
    assert np.allclose(
        results[0].scores.cpu().numpy(), np.array([0.8682, 0.7068, 0.4445]), atol=0.01
    )
    assert 724600 <= results[0].masks[0].cpu().sum() <= 725000
    assert np.allclose(
        results[1].scores.cpu().numpy(), np.array([0.8682, 0.7068, 0.4445]), atol=0.01
    )
    assert 724600 <= results[1].masks[0].cpu().sum() <= 725000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_without_prompting_torch(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)

    # when
    results = model.segment_images(truck_image_torch)

    # then
    assert len(results) == 1
    assert np.allclose(
        results[0].scores.cpu().numpy(), np.array([0.8682, 0.7068, 0.4445]), atol=0.01
    )
    assert 724600 <= results[0].masks[0].cpu().sum() <= 725000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_without_prompting_batch_torch(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)

    # when
    results = model.segment_images([truck_image_torch, truck_image_torch])

    # then
    assert len(results) == 2
    assert np.allclose(
        results[0].scores.cpu().numpy(), np.array([0.8682, 0.7068, 0.4445]), atol=0.01
    )
    assert 724600 <= results[0].masks[0].cpu().sum() <= 725000
    assert np.allclose(
        results[1].scores.cpu().numpy(), np.array([0.8682, 0.7068, 0.4445]), atol=0.01
    )
    assert 724600 <= results[1].masks[0].cpu().sum() <= 725000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_points_prompting(
    sam_package: str,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    # when
    results = model.segment_images(
        truck_image_numpy,
        point_coordinates=input_point,
        point_labels=input_label,
    )

    # then
    assert len(results) == 1
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([0.9052551, 0.95114934, 0.9838573]),
        atol=0.01,
    )
    assert 21500 <= results[0].masks[2].cpu().sum() <= 21700


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_points_prompting_for_multiple_points(
    sam_package: str,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_point = np.array([[500, 375], [500, 375], [500, 375]])
    input_label = np.array([1, 1, 1])

    # when
    results = model.segment_images(
        truck_image_numpy,
        point_coordinates=input_point,
        point_labels=input_label,
    )

    # then
    assert len(results) == 1
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([0.8865736, 0.98261404, 0.98639965]),
        atol=0.01,
    )
    assert 21500 <= results[0].masks[2].cpu().sum() <= 21700


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_points_prompting_with_embeddings(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    # when
    embeddings = model.embed_images(truck_image_torch)
    results = model.segment_images(
        embeddings=[embeddings[0], embeddings[0]],
        point_coordinates=input_point,
        point_labels=input_label,
    )

    # then
    assert len(results) == 2
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([0.9052551, 0.95114934, 0.9838573]),
        atol=0.01,
    )
    assert 21500 <= results[0].masks[2].cpu().sum() <= 21700
    assert np.allclose(
        results[1].scores.cpu().numpy(),
        np.array([0.9052551, 0.95114934, 0.9838573]),
        atol=0.01,
    )
    assert 21500 <= results[1].masks[2].cpu().sum() <= 21700


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_box_prompting(
    sam_package: str,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_box = np.array([425, 600, 700, 875])

    # when
    results = model.segment_images(
        truck_image_numpy,
        boxes=input_box,
    )

    # then
    assert len(results) == 1
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([0.97505486, 0.96572363, 0.9098512]),
        atol=0.01,
    )
    assert 31300 <= results[0].masks[2].cpu().sum() <= 31500


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_box_prompting_and_embeddings(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_box = np.array([425, 600, 700, 875])

    # when
    embeddings = model.embed_images(truck_image_torch)
    results = model.segment_images(
        embeddings=[embeddings[0], embeddings[0]],
        boxes=input_box,
    )

    # then
    assert len(results) == 2
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([0.97505486, 0.96572363, 0.9098512]),
        atol=0.01,
    )
    assert 31300 <= results[0].masks[2].cpu().sum() <= 31500
    assert np.allclose(
        results[1].scores.cpu().numpy(),
        np.array([0.97505486, 0.96572363, 0.9098512]),
        atol=0.01,
    )
    assert 31300 <= results[1].masks[2].cpu().sum() <= 31500


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_mask_prompting(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_box = np.array([425, 600, 700, 875])

    # when
    first_results = model.segment_images(
        images=truck_image_torch,
    )
    second_results = model.segment_images(
        images=truck_image_torch,
        mask_input=first_results[0].logits[0],
        input_box=input_box,
    )

    # then
    assert len(second_results) == 1
    assert np.allclose(
        second_results[0].scores.cpu().numpy(),
        np.array([0.69966555, 0.5946327, 0.33864248]),
        atol=0.01,
    )
    assert 1200 <= second_results[0].masks[2].cpu().sum() <= 1400


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_combined_prompting(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_box = torch.from_numpy(np.array([425, 600, 700, 875]))
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    # when
    results = model.segment_images(
        truck_image_torch,
        point_coordinates=[input_point],
        point_labels=[input_label],
        boxes=input_box,
    )

    # then
    assert len(results) == 1
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([0.93800557, 0.87928826, 0.7825271]),
        atol=0.01,
    )
    assert 38600 <= results[0].masks[0].cpu().sum() <= 39000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_batch_size_misaligned_prompts_batch_size(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_box = torch.from_numpy(np.array([[425, 600, 700, 875]]))
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    # when
    with pytest.raises(ModelInputError):
        _ = model.segment_images(
            truck_image_torch,
            point_coordinates=[input_point],
            point_labels=[input_label, input_label],
            boxes=input_box,
        )


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_batch_size_misaligned_prompts_size_structure(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_box = torch.from_numpy(
        np.array(
            [
                [425, 600, 700, 875],
                [425, 600, 700, 875],
            ]
        )
    )
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    # when
    with pytest.raises(ModelInputError):
        _ = model.segment_images(
            truck_image_torch,
            point_coordinates=[input_point],
            point_labels=[input_label],
            boxes=input_box,
        )


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_multiple_prompt_elements_for_single_image(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_box = torch.from_numpy(
        np.array(
            [
                [425, 600, 700, 875],
                [425, 600, 700, 875],
            ]
        )
    )

    # when
    results = model.segment_images(
        truck_image_torch,
        boxes=input_box,
        multi_mask_output=True,
    )

    # then
    assert len(results) == 1
    assert results[0].masks.shape == (2, 3, 1200, 1800)


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_multiple_prompt_elements_for_multiple_images(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_box = torch.from_numpy(
        np.array(
            [
                [425, 600, 700, 875],
                [425, 600, 700, 875],
            ]
        )
    )

    # when
    results = model.segment_images(
        [truck_image_torch, truck_image_torch],
        boxes=[input_box, input_box],
        multi_mask_output=True,
    )

    # then
    assert len(results) == 2
    assert results[0].masks.shape == (2, 3, 1200, 1800)
    assert results[1].masks.shape == (2, 3, 1200, 1800)


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam_predictions_with_multiple_mixed_prompt_elements_for_single_image(
    sam_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAMTorch.from_pretrained(sam_package, device=DEFAULT_DEVICE)
    input_box = torch.from_numpy(
        np.array(
            [
                [425, 600, 700, 875],
                [425, 600, 700, 875],
            ]
        )
    )
    input_point = np.array([[575, 750], [575, 750]])
    input_label = np.array([0, 1])

    # when
    with pytest.raises(ModelInputError):
        _ = model.segment_images(
            truck_image_torch,
            point_coordinates=input_point,
            point_labels=input_label,
            boxes=input_box,
            multi_mask_output=True,
        )
