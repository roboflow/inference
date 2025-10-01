import cv2
import numpy as np
import pytest
import torch
from clip.clip import _transform
from inference_exp.errors import ModelRuntimeError
from inference_exp.models.clip.preprocessing import create_clip_preprocessor
from PIL import Image


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_single_numpy_bgr_input(
    dog_image_numpy: np.ndarray, dog_image_pil: Image.Image
) -> None:
    # given
    original_preprocessor = _transform(n_px=224)
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    ours_output = inference_preprocessor(dog_image_numpy, None, torch.device("cpu"))
    original_output = original_preprocessor(dog_image_pil).unsqueeze(0)

    # then
    max_diff = torch.max(torch.abs(ours_output - original_output))
    assert max_diff < 0.03


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_single_numpy_rgb_input(
    dog_image_numpy: np.ndarray, dog_image_pil: Image.Image
) -> None:
    # given
    original_preprocessor = _transform(n_px=224)
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    ours_output_rgb = inference_preprocessor(
        np.ascontiguousarray(dog_image_numpy[:, :, ::-1]), "rgb", torch.device("cpu")
    )
    ours_output_bgr = inference_preprocessor(dog_image_numpy, None, torch.device("cpu"))
    original_output = original_preprocessor(dog_image_pil).unsqueeze(0)

    # then
    assert torch.allclose(ours_output_rgb, ours_output_bgr)
    max_diff = torch.max(torch.abs(ours_output_rgb - original_output))
    assert max_diff < 0.03


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_single_3d_rgb_tensor_input(
    dog_image_torch: torch.Tensor,
    dog_image_numpy: np.ndarray,
    dog_image_pil: Image.Image,
) -> None:
    # given
    original_preprocessor = _transform(n_px=224)
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    ours_output_torch = inference_preprocessor(
        dog_image_torch, None, torch.device("cpu")
    )
    ours_output_numpy = inference_preprocessor(
        dog_image_torch, None, torch.device("cpu")
    )
    original_output = original_preprocessor(dog_image_pil).unsqueeze(0)

    # then
    assert torch.allclose(ours_output_torch, ours_output_numpy)
    max_diff = torch.max(torch.abs(ours_output_torch - original_output))
    assert max_diff < 0.03


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_list_of_numpy_inputs_bgr(
    dog_image_numpy: np.ndarray, dog_image_pil: Image.Image
) -> None:
    # given
    original_preprocessor = _transform(n_px=224)
    inference_preprocessor = create_clip_preprocessor(image_size=224)
    input_images = [dog_image_numpy, dog_image_numpy]

    # when
    ours_output = inference_preprocessor(input_images, None, torch.device("cpu"))
    original_output = original_preprocessor(dog_image_pil).unsqueeze(0)

    # then
    assert ours_output.shape[0] == 2
    assert torch.max(torch.abs(ours_output[0] - original_output[0])) < 0.03
    assert torch.max(torch.abs(ours_output[1] - original_output[0])) < 0.03


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_list_of_numpy_inputs_rgb(
    dog_image_numpy: np.ndarray, dog_image_pil: Image.Image
) -> None:
    # given
    original_preprocessor = _transform(n_px=224)
    inference_preprocessor = create_clip_preprocessor(image_size=224)
    input_images = [
        np.ascontiguousarray(dog_image_numpy[:, :, ::-1]),
        np.ascontiguousarray(dog_image_numpy[:, :, ::-1]),
    ]

    # when
    ours_output = inference_preprocessor(input_images, "rgb", torch.device("cpu"))
    original_output = original_preprocessor(dog_image_pil).unsqueeze(0)

    # then
    assert ours_output.shape[0] == 2
    assert torch.max(torch.abs(ours_output[0] - original_output[0])) < 0.03
    assert torch.max(torch.abs(ours_output[1] - original_output[0])) < 0.03


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_list_of_tensor_inputs(
    dog_image_torch: torch.Tensor, dog_image_pil: Image.Image
) -> None:
    # given
    original_preprocessor = _transform(n_px=224)
    inference_preprocessor = create_clip_preprocessor(image_size=224)
    input_images = [dog_image_torch, dog_image_torch]

    # when
    ours_output = inference_preprocessor(input_images, None, torch.device("cpu"))
    original_output = original_preprocessor(dog_image_pil).unsqueeze(0)

    # then
    assert ours_output.shape[0] == 2
    assert torch.max(torch.abs(ours_output[0] - original_output[0])) < 0.03
    assert torch.max(torch.abs(ours_output[1] - original_output[0])) < 0.03


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_list_of_elements_of_different_type(
    dog_image_torch: torch.Tensor,
    dog_image_numpy: np.ndarray,
    dog_image_pil: Image.Image,
) -> None:
    # given
    original_preprocessor = _transform(n_px=224)
    inference_preprocessor = create_clip_preprocessor(image_size=224)
    input_images = [dog_image_torch, dog_image_numpy]

    # when
    ours_output = inference_preprocessor(input_images, None, torch.device("cpu"))
    original_output = original_preprocessor(dog_image_pil).unsqueeze(0)

    # then
    assert ours_output.shape[0] == 2
    assert torch.max(torch.abs(ours_output[0] - original_output[0])) < 0.03
    assert torch.max(torch.abs(ours_output[1] - original_output[0])) < 0.03


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_list_of_elements_of_different_shape(
    dog_image_numpy: np.ndarray, dog_image_pil: Image.Image
) -> None:
    # given
    original_preprocessor = _transform(n_px=224)
    inference_preprocessor = create_clip_preprocessor(image_size=224)
    input_images = [dog_image_numpy, cv2.resize(dog_image_numpy, (1920, 1080))]

    # when
    ours_output = inference_preprocessor(input_images, None, torch.device("cpu"))
    original_output = original_preprocessor(dog_image_pil).unsqueeze(0)

    # then
    assert ours_output.shape[0] == 2
    assert torch.max(torch.abs(ours_output[0] - original_output[0])) < 0.03


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_batched_tensor_input(
    dog_image_torch: torch.Tensor, dog_image_pil: Image.Image
) -> None:
    # given
    original_preprocessor = _transform(n_px=224)
    inference_preprocessor = create_clip_preprocessor(image_size=224)
    input_images = torch.stack([dog_image_torch, dog_image_torch], dim=0)

    # when
    ours_output = inference_preprocessor(input_images, None, torch.device("cpu"))
    original_output = original_preprocessor(dog_image_pil).unsqueeze(0)

    # then
    assert ours_output.shape[0] == 2
    assert torch.max(torch.abs(ours_output[0] - original_output[0])) < 0.03
    assert torch.max(torch.abs(ours_output[1] - original_output[0])) < 0.03


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_when_empty_list_provided() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor([], None, torch.device("cpu"))


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_when_string_provided() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor("some", None, torch.device("cpu"))


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_when_list_of_string_provided() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor(["some"], None, torch.device("cpu"))


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_when_single_channel_np_array_provided() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor(np.zeros((192, 168, 1)), None, torch.device("cpu"))


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_when_invalid_channel_np_array_provided() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor(np.zeros((192, 168, 4)), None, torch.device("cpu"))


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_when_invalid_channel_np_array_provided_in_list() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor([np.zeros((192, 168, 1))], None, torch.device("cpu"))


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_when_batched_np_array_provided() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor(
            np.zeros((3, 192, 168, 3)), None, torch.device("cpu")
        )


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_when_batched_np_array_provided_in_list() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor(
            [np.zeros((3, 192, 168, 3))], None, torch.device("cpu")
        )


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_invalid_channel_tensor_provided() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor(
            torch.zeros(size=(1, 192, 168)), None, torch.device("cpu")
        )


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_invalid_channel_tensor_provided_in_list() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor(
            [torch.zeros(size=(1, 192, 168))], None, torch.device("cpu")
        )


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_clip_preprocessor_batched_tensor_provided_in_list() -> None:
    # given
    inference_preprocessor = create_clip_preprocessor(image_size=224)

    # when
    with pytest.raises(ModelRuntimeError):
        _ = inference_preprocessor(
            [torch.zeros(size=(10, 3, 192, 168))], None, torch.device("cpu")
        )
