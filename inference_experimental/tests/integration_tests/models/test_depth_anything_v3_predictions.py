import numpy as np
import pytest
import torch
from inference_exp.configuration import DEFAULT_DEVICE


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_for_numpy_image(
    depth_anything_v3_small_package: str, dog_image_numpy: np.ndarray
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_small_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model(dog_image_numpy)

    # then
    assert len(results) == 1
    assert results[0].cpu().shape == (1280, 720)


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_for_numpy_images_list(
    depth_anything_v3_small_package: str, dog_image_numpy: np.ndarray
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_small_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model([dog_image_numpy, dog_image_numpy])

    # then
    assert len(results) == 2
    assert results[0].cpu().shape == (1280, 720)
    assert results[1].cpu().shape == (1280, 720)


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_for_torch_image(
    depth_anything_v3_small_package: str, dog_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_small_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model(dog_image_torch)

    # then
    assert len(results) == 1
    assert results[0].cpu().shape == (1280, 720)


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_for_torch_batch(
    depth_anything_v3_small_package: str, dog_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_small_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model(torch.stack([dog_image_torch, dog_image_torch], dim=0))

    # then
    assert len(results) == 2
    assert results[0].cpu().shape == (1280, 720)
    assert results[1].cpu().shape == (1280, 720)


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_for_torch_list(
    depth_anything_v3_small_package: str, dog_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_small_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model([dog_image_torch, dog_image_torch])

    # then
    assert len(results) == 2
    assert results[0].cpu().shape == (1280, 720)
    assert results[1].cpu().shape == (1280, 720)


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_base_for_numpy_image(
    depth_anything_v3_base_package: str, dog_image_numpy: np.ndarray
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_base_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model(dog_image_numpy)

    # then
    assert len(results) == 1
    assert results[0].cpu().shape == (1280, 720)


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_base_for_numpy_images_list(
    depth_anything_v3_base_package: str, dog_image_numpy: np.ndarray
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_base_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model([dog_image_numpy, dog_image_numpy])

    # then
    assert len(results) == 2
    assert results[0].cpu().shape == (1280, 720)
    assert results[1].cpu().shape == (1280, 720)


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_base_for_torch_image(
    depth_anything_v3_base_package: str, dog_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_base_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model(dog_image_torch)

    # then
    assert len(results) == 1
    assert results[0].cpu().shape == (1280, 720)


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_base_for_torch_batch(
    depth_anything_v3_base_package: str, dog_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_base_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model(torch.stack([dog_image_torch, dog_image_torch], dim=0))

    # then
    assert len(results) == 2
    assert results[0].cpu().shape == (1280, 720)
    assert results[1].cpu().shape == (1280, 720)


@pytest.mark.slow
@pytest.mark.torch_models
def test_depth_anything_v3_base_for_torch_list(
    depth_anything_v3_base_package: str, dog_image_torch: torch.Tensor
) -> None:
    # given
    from inference_exp.models.depth_anything_v3.depth_anything_v3_torch import (
        DepthAnythingV3Torch,
    )

    model = DepthAnythingV3Torch.from_pretrained(
        depth_anything_v3_base_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = model([dog_image_torch, dog_image_torch])

    # then
    assert len(results) == 2
    assert results[0].cpu().shape == (1280, 720)
    assert results[1].cpu().shape == (1280, 720)

