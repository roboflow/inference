import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.dominant_color.v1 import (
    DominantColorBlockV1,
    DominantColorManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_dominant_color_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/dominant_color@v1",
        "name": "dominant_color1",
        "images": "$inputs.image",
    }

    # when
    result = DominantColorManifest.model_validate(data)

    # then
    assert result == DominantColorManifest(
        type="roboflow_core/dominant_color@v1",
        name="dominant_color1",
        images="$inputs.image",
        color_clusters=4,
        max_iterations=100,
    ), "Expected the manifest to be validated successfully with default values"


def test_dominant_color_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "DominantColor",
        "name": "dominant_color1",
        "images": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = DominantColorManifest.model_validate(data)


def test_dominant_color_block() -> None:
    # given
    block = DominantColorBlockV1()

    # generate a red image to test
    red_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    red_image[:, :, 2] = 255

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=red_image,
        ),
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    assert output is not None, "Expected an output but got None"
    assert isinstance(output["rgb_color"], tuple), " Expected rgb_color to be a tuple"
    assert len(output["rgb_color"]) == 3, " Expected rgb_color to have 3 elements"
    assert all(
        0 <= color <= 255 for color in output["rgb_color"]
    ), " Expected all elements in rgb_color to be between 0 and 255"
    assert output["rgb_color"] == (
        255,
        0,
        0,
    ), " Expected rgb_color to be [255, 0, 0], aka a red image"


# --- tensor-native sibling ---------------------------------------------------
# Parity contract: the sibling downsamples ON DEVICE, then hands the host a
# BGR HWC array byte-identical to v1's numpy_image[::s, ::s] and runs the very
# same numpy k-means (find_dominant_color, imported from v1). Identical bytes
# + identical global-RNG state force an identical clustering trajectory, so
# outputs must be EXACTLY equal. v1 is unseeded (nondeterministic run-to-run),
# hence every parity test pins np.random.seed immediately before each run.


def _tensor_dominant_color_imports():
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.classical_cv.dominant_color import (
        v1_tensor,
    )

    return torch, v1_tensor


def _paired_images(torch, bgr: np.ndarray):
    numpy_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=bgr,
    )
    chw = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).contiguous()
    tensor_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        tensor_image=chw,
    )
    return numpy_born, tensor_born


def _dominant_case_image(case: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    if case == "noise":
        return rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    if case == "dominant_red":
        image = np.zeros((250, 300, 3), dtype=np.uint8)
        image[:, :, 2] = 255  # BGR red
        image[rng.random((250, 300)) < 0.1] = (0, 255, 0)  # ~10% green speckle
        return image
    if case == "non_divisible":
        # 317x203: strides do not divide the dims evenly
        return rng.integers(0, 256, size=(317, 203, 3), dtype=np.uint8)
    raise ValueError(case)


@pytest.mark.parametrize("seed", [7, 1234])
@pytest.mark.parametrize("case", ["noise", "dominant_red", "non_divisible"])
def test_tensor_dominant_color_seeded_parity(seed: int, case: str) -> None:
    # given - the same pixels as numpy-born BGR and tensor-born RGB CHW
    torch, v1_tensor = _tensor_dominant_color_imports()
    bgr = _dominant_case_image(case)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when - identical global-RNG state before each run forces identical draws
    np.random.seed(seed)
    numpy_result = DominantColorBlockV1().run(
        image=numpy_born, color_clusters=4, max_iterations=100, target_size=100
    )
    np.random.seed(seed)
    tensor_result = v1_tensor.DominantColorBlockV1().run(
        image=tensor_born, color_clusters=4, max_iterations=100, target_size=100
    )

    # then - byte-identical downsample + identical RNG => identical trajectory
    assert isinstance(tensor_result["rgb_color"], tuple)
    assert tensor_result["rgb_color"] == numpy_result["rgb_color"]


def test_tensor_dominant_color_seeded_parity_with_non_default_parameters() -> None:
    # given
    torch, v1_tensor = _tensor_dominant_color_imports()
    bgr = _dominant_case_image("non_divisible")
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when
    np.random.seed(97)
    numpy_result = DominantColorBlockV1().run(
        image=numpy_born, color_clusters=6, max_iterations=30, target_size=64
    )
    np.random.seed(97)
    tensor_result = v1_tensor.DominantColorBlockV1().run(
        image=tensor_born, color_clusters=6, max_iterations=30, target_size=64
    )

    # then
    assert tensor_result["rgb_color"] == numpy_result["rgb_color"]


@pytest.mark.parametrize("target_size", [250, 100, 50])
def test_tensor_dominant_color_downsample_matches_v1_slicing(target_size: int) -> None:
    # given - 317x203 exercises strided slicing with a remainder
    torch, v1_tensor = _tensor_dominant_color_imports()
    bgr = _dominant_case_image("non_divisible")
    numpy_born, tensor_born = _paired_images(torch, bgr)
    height, width = bgr.shape[:2]
    scale_factor = max(1, min(width, height) // target_size)

    # when
    device_downsampled = v1_tensor._downsample_to_bgr_numpy(
        chw=tensor_born.tensor_image, scale_factor=scale_factor
    )

    # then - byte-identical to v1's strided numpy slice: THE invariant that
    # keeps the k-means trajectory shared between the two paths
    assert scale_factor == {250: 1, 100: 2, 50: 4}[target_size]
    assert device_downsampled.dtype == np.uint8
    assert np.array_equal(
        device_downsampled, numpy_born.numpy_image[::scale_factor, ::scale_factor]
    )


def test_tensor_dominant_color_numpy_born_delegates_without_materialisation() -> None:
    # given
    torch, v1_tensor = _tensor_dominant_color_imports()
    bgr = _dominant_case_image("noise")
    numpy_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=bgr,
    )
    np.random.seed(3)
    reference = DominantColorBlockV1().run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="parent"),
            numpy_image=bgr,
        ),
        color_clusters=4,
        max_iterations=100,
        target_size=100,
    )

    # when
    np.random.seed(3)
    result = v1_tensor.DominantColorBlockV1().run(
        image=numpy_born, color_clusters=4, max_iterations=100, target_size=100
    )

    # then - identical output via the numpy delegate, and no forced H2D
    assert result["rgb_color"] == reference["rgb_color"]
    assert not numpy_born.is_tensor_materialised(), "delegate must not materialise"


def test_tensor_dominant_color_obvious_dominant_color_on_tensor_path() -> None:
    # given - ~90% red / ~10% green speckle, tensor-born
    torch, v1_tensor = _tensor_dominant_color_imports()
    bgr = _dominant_case_image("dominant_red")
    _, tensor_born = _paired_images(torch, bgr)

    # when
    np.random.seed(11)
    result = v1_tensor.DominantColorBlockV1().run(
        image=tensor_born, color_clusters=4, max_iterations=100, target_size=100
    )

    # then - dominant colour is red-ish (greens may or may not split off into
    # their own cluster depending on the seeded init, hence bounds, not equality)
    r, g, b = result["rgb_color"]
    assert r >= 200
    assert g <= 60
    assert b <= 60


def test_tensor_dominant_color_insufficient_pixels_raises_on_both_paths() -> None:
    # given - 2x2 image: 4 downsampled pixels < 10 requested clusters, so
    # np.random.choice(..., replace=False) must raise on BOTH paths (error
    # parity: the sibling does not "fix" v1 edge cases)
    torch, v1_tensor = _tensor_dominant_color_imports()
    bgr = np.full((2, 2, 3), 128, dtype=np.uint8)
    numpy_born, tensor_born = _paired_images(torch, bgr)

    # when / then
    with pytest.raises(ValueError):
        DominantColorBlockV1().run(
            image=numpy_born, color_clusters=10, max_iterations=100, target_size=100
        )
    with pytest.raises(ValueError):
        v1_tensor.DominantColorBlockV1().run(
            image=tensor_born, color_clusters=10, max_iterations=100, target_size=100
        )


def test_tensor_dominant_color_on_mps_device(monkeypatch) -> None:
    # given - tensor images live on the globally configured device; simulate an
    # MPS deployment by patching the global, then check the device-side
    # downsample + host k-means still match v1 exactly under a pinned RNG
    torch, v1_tensor = _tensor_dominant_color_imports()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    import inference.core.workflows.execution_engine.entities.base as base_module

    bgr = _dominant_case_image("non_divisible")
    numpy_born, _ = _paired_images(torch, bgr)
    np.random.seed(21)
    reference = DominantColorBlockV1().run(
        image=numpy_born, color_clusters=4, max_iterations=100, target_size=100
    )
    monkeypatch.setattr(base_module, "WORKFLOWS_IMAGE_TENSOR_DEVICE", "mps")
    _, tensor_born = _paired_images(torch, bgr)
    assert tensor_born.tensor_image.device.type == "mps"

    # when
    np.random.seed(21)
    result = v1_tensor.DominantColorBlockV1().run(
        image=tensor_born, color_clusters=4, max_iterations=100, target_size=100
    )

    # then - device slice on MPS, identical result
    assert result["rgb_color"] == reference["rgb_color"]
