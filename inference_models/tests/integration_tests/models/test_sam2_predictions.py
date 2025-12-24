import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import ModelInputError
from inference_models.models.sam2.sam2_torch import SAM2Torch


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_embeddings_numpy(
    sam2_package: str, truck_image_numpy: np.ndarray
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)

    # when
    results = model.embed_images(truck_image_numpy)

    # then
    assert len(results) == 1
    assert 14200 <= results[0].embeddings.cpu().sum() <= 14800


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_embeddings_torch(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)

    # when
    results = model.embed_images(truck_image_torch)

    # then
    assert len(results) == 1
    assert 14200 <= results[0].embeddings.cpu().sum() <= 14800


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_without_prompting_numpy(
    sam2_package: str, truck_image_numpy: np.ndarray
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)

    # when
    results = model.segment_images(truck_image_numpy)

    # then
    assert len(results) == 1
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([5.454666e-09, 5.505157e-10, 6.009522e-10]),
        atol=0.0001,
    )
    assert results[0].masks[0].cpu().sum() <= 1000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_without_prompting_batch_numpy(
    sam2_package: str, truck_image_numpy: np.ndarray
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)

    # when
    results = model.segment_images([truck_image_numpy, truck_image_numpy])

    # then
    assert len(results) == 2
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([5.454666e-09, 5.505157e-10, 6.009522e-10]),
        atol=0.0001,
    )
    assert results[0].masks[0].cpu().sum() <= 1000
    assert np.allclose(
        results[1].scores.cpu().numpy(),
        np.array([5.454666e-09, 5.505157e-10, 6.009522e-10]),
        atol=0.0001,
    )
    assert results[1].masks[0].cpu().sum() <= 1000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_without_prompting_torch(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)

    # when
    results = model.segment_images(truck_image_torch)

    # then
    assert len(results) == 1
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([5.454666e-09, 5.505157e-10, 6.009522e-10]),
        atol=0.0001,
    )
    assert results[0].masks[0].cpu().sum() <= 1200


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_without_prompting_batch_torch(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)

    # when
    results = model.segment_images([truck_image_torch, truck_image_torch])

    # then
    assert len(results) == 2
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([5.454666e-09, 5.505157e-10, 6.009522e-10]),
        atol=0.0001,
    )
    assert results[0].masks[0].cpu().sum() <= 1200
    assert np.allclose(
        results[1].scores.cpu().numpy(),
        np.array([5.454666e-09, 5.505157e-10, 6.009522e-10]),
        atol=0.0001,
    )
    assert results[1].masks[0].cpu().sum() <= 1200


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_points_prompting(
    sam2_package: str,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
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
        np.array([0.03849015, 0.8936804, 0.54487956]),
        atol=0.01,
    )
    assert 46500 <= results[0].masks[2].cpu().sum() <= 47500


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_points_prompting_for_multiple_points(
    sam2_package: str,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
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
        np.array([0.04465561, 0.93181133, 0.5375561]),
        atol=0.01,
    )
    assert 20000 <= results[0].masks[1].cpu().sum() <= 21000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_points_prompting_with_embeddings(
    sam2_package: str, truck_image_numpy: np.ndarray
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    # when
    embeddings = model.embed_images(truck_image_numpy)
    results = model.segment_images(
        embeddings=[embeddings[0], embeddings[0]],
        point_coordinates=input_point,
        point_labels=input_label,
    )

    # then
    assert len(results) == 2
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([0.03849015, 0.8936804, 0.54487956]),
        atol=0.01,
    )
    assert 46500 <= results[1].masks[2].cpu().sum() <= 47500
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([0.03849015, 0.8936804, 0.54487956]),
        atol=0.01,
    )
    assert 46500 <= results[1].masks[2].cpu().sum() <= 47500


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_box_prompting(
    sam2_package: str,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
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
        np.array([[0.6037196, 0.9654025, 0.9389837]]),
        atol=0.01,
    )
    assert 42000 <= results[0].masks[1].cpu().sum() <= 42600


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_box_prompting_and_embeddings(
    sam2_package: str,
    truck_image_numpy: np.ndarray,
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
    input_box = np.array([425, 600, 700, 875])

    # when
    embeddings = model.embed_images(truck_image_numpy)
    results = model.segment_images(
        embeddings=[embeddings[0], embeddings[0]],
        boxes=input_box,
    )

    # then
    assert len(results) == 2
    assert np.allclose(
        results[0].scores.cpu().numpy(),
        np.array([[0.6037196, 0.9654025, 0.9389837]]),
        atol=0.01,
    )
    assert 42000 <= results[0].masks[1].cpu().sum() <= 42600
    assert np.allclose(
        results[1].scores.cpu().numpy(),
        np.array([[0.6037196, 0.9654025, 0.9389837]]),
        atol=0.01,
    )
    assert 42000 <= results[1].masks[1].cpu().sum() <= 42600


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_mask_prompting(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
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
        np.array([1.6569568e-10, 1.5815127e-11, 3.0368791e-11]),
        atol=0.0001,
    )
    assert second_results[0].masks[2].cpu().sum() <= 3000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_combined_prompting(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    # given
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
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
        np.array([0.40400937, 0.9135251, 0.92012656]),
        atol=0.01,
    )
    assert 79000 <= results[0].masks[0].cpu().sum() <= 80000


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_batch_size_misaligned_prompts_batch_size(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
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
def test_sam2_predictions_with_batch_size_misaligned_prompts_size_structure(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
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
def test_sam2_predictions_with_multiple_prompt_elements_for_single_image(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
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
def test_sam2_predictions_with_multiple_prompt_elements_for_multiple_images(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
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
def test_sam2_predictions_with_multiple_mixed_prompt_elements_for_single_image(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
    input_box = torch.from_numpy(np.array([[425, 600, 700, 875]]))
    input_point = np.array([[575, 750], [575, 750]])
    input_label = np.array([0, 1])

    # when
    results = model.segment_images(
        truck_image_torch,
        point_coordinates=input_point,
        point_labels=input_label,
        boxes=input_box,
        multi_mask_output=True,
    )

    assert len(results) == 1
    assert results[0].masks.shape == (3, 1200, 1800)


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_multiple_prompt_elements_for_single_image(
    sam2_package: str, truck_image_torch: torch.Tensor
) -> None:
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
    input_box = torch.from_numpy(np.array([[425, 600, 700, 875], [425, 600, 700, 875]]))
    input_point = np.array([[[575, 750], [575, 750]], [[575, 750], [575, 750]]])
    input_label = np.array([[0, 1], [0, 1]])

    # when
    results = model.segment_images(
        truck_image_torch,
        point_coordinates=input_point,
        point_labels=input_label,
        boxes=input_box,
        multi_mask_output=True,
    )

    assert len(results) == 1
    assert results[0].masks.shape == (2, 3, 1200, 1800)


@pytest.mark.slow
@pytest.mark.torch_models
def test_sam2_predictions_with_multiple_prompt_elements_for_multiple_images(
    sam2_package: str,
    truck_image_torch: torch.Tensor,
    truck_image_numpy: np.ndarray,
) -> None:
    model = SAM2Torch.from_pretrained(sam2_package, device=DEFAULT_DEVICE)
    input_box_1 = torch.from_numpy(
        np.array([[425, 600, 700, 875], [425, 600, 700, 875]])
    )
    input_point_1 = np.array([[[575, 750], [575, 750]], [[575, 750], [575, 750]]])
    input_label_1 = np.array([[0, 1], [0, 1]])

    input_box_2 = torch.from_numpy(
        np.array([[425, 600, 700, 875], [425, 600, 700, 875], [425, 600, 700, 875]])
    )
    input_point_2 = np.array(
        [
            [[575, 750], [575, 750], [575, 750]],
            [[575, 750], [575, 750], [575, 750]],
            [[575, 750], [575, 750], [575, 750]],
        ]
    )
    input_label_2 = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 1]])

    # when
    results = model.segment_images(
        [truck_image_torch, truck_image_numpy],
        point_coordinates=[input_point_1, input_point_2],
        point_labels=[input_label_1, input_label_2],
        boxes=[input_box_1, input_box_2],
        multi_mask_output=True,
    )

    assert len(results) == 2
    assert results[0].masks.shape == (2, 3, 1200, 1800)
    assert results[1].masks.shape == (3, 3, 1200, 1800)
