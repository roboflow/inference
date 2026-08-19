import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.configuration import (
    INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.errors import MissingDependencyError, ModelInputError
from inference_models.models.mage_vl.mage_vl_hf import (
    CODEC_PATCH_SIZE,
    MageVLHF,
    _get_mage_vl_attn_implementation,
)


def _mage_vl(processor: MagicMock = None, model: MagicMock = None) -> MageVLHF:
    model = model or MagicMock()
    model.dtype = torch.bfloat16
    processor = processor or MagicMock()
    processor.apply_chat_template.return_value = "<|im_start|>user"
    processor.return_value = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
        "pixel_values": torch.tensor([[[1.0]]], dtype=torch.float32),
    }
    return MageVLHF(
        model=model,
        processor=processor,
        model_package_dir="/model/package",
        device=torch.device("cpu"),
    )


def test_pre_process_generation_flips_bgr_numpy_images_to_rgb() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.return_value = {"input_ids": torch.tensor([[1]], dtype=torch.int64)}
    mage_vl = _mage_vl(processor=processor)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue channel of a BGR image

    mage_vl.pre_process_generation(images=image, prompt="what is this?")

    passed_image = processor.call_args.kwargs["images"]
    assert passed_image[..., 2].max() == 255, "blue must end up in the last channel"
    assert passed_image[..., 0].max() == 0


def test_pre_process_generation_leaves_rgb_numpy_images_alone() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.return_value = {"input_ids": torch.tensor([[1]], dtype=torch.int64)}
    mage_vl = _mage_vl(processor=processor)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 255

    mage_vl.pre_process_generation(
        images=image, prompt="what is this?", input_color_format="rgb"
    )

    passed_image = processor.call_args.kwargs["images"]
    assert passed_image[..., 0].max() == 255


def test_pre_process_generation_casts_pixel_values_to_model_dtype() -> None:
    mage_vl = _mage_vl()

    inputs = mage_vl.pre_process_generation(images=np.zeros((8, 8, 3), dtype=np.uint8))

    assert inputs["input_ids"].dtype == torch.int64
    assert inputs["pixel_values"].dtype == torch.bfloat16


def test_pre_process_generation_rejects_no_media() -> None:
    mage_vl = _mage_vl()

    with pytest.raises(ModelInputError):
        mage_vl.pre_process_generation(prompt="describe")


def test_pre_process_generation_rejects_both_image_and_video(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    mage_vl = _mage_vl()

    with pytest.raises(ModelInputError):
        mage_vl.pre_process_generation(
            images=np.zeros((8, 8, 3), dtype=np.uint8), video=str(video)
        )


def test_pre_process_generation_rejects_missing_video_file() -> None:
    mage_vl = _mage_vl()

    with pytest.raises(ModelInputError):
        mage_vl.pre_process_generation(video="/does/not/exist.mp4")


def test_pre_process_generation_rejects_unknown_codec_engine(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    mage_vl = _mage_vl()

    with pytest.raises(ModelInputError):
        mage_vl.pre_process_generation(video=str(video), codec_engine="mpeg-2")


def test_pre_process_generation_builds_hevc_codec_config(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.return_value = {"input_ids": torch.tensor([[1]], dtype=torch.int64)}
    mage_vl = _mage_vl(processor=processor)

    mage_vl.pre_process_generation(
        video=str(video), codec_engine="hevc", target_canvas=12, max_pixels=1234
    )

    kwargs = processor.call_args.kwargs
    assert kwargs["videos"] == [str(video)]
    assert kwargs["video_backend"] == "codec"
    assert kwargs["max_pixels"] == 1234
    assert kwargs["codec_config"] == {
        "engine": "hevc",
        "target_canvas": 12,
        "patch": CODEC_PATCH_SIZE,
    }


def test_pre_process_generation_points_dcvc_at_the_bundled_codec(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.return_value = {"input_ids": torch.tensor([[1]], dtype=torch.int64)}
    mage_vl = _mage_vl(processor=processor)

    mage_vl.pre_process_generation(video=str(video), codec_engine="dcvc-rt")

    codec_config = processor.call_args.kwargs["codec_config"]
    assert codec_config["engine"] == "dcvc-rt"
    assert codec_config["dcvc"]["pkg_dir"] == os.path.join(
        "/model/package", "neural_codec"
    )
    assert codec_config["dcvc"]["device"] == "cpu"


def test_generate_returns_only_newly_generated_tokens() -> None:
    model = MagicMock()
    model.dtype = torch.bfloat16
    model.generate.return_value = torch.tensor([[11, 12, 21, 22]])
    mage_vl = _mage_vl(model=model)

    result = mage_vl.generate(inputs={"input_ids": torch.tensor([[11, 12]])})

    assert result.tolist() == [[21, 22]]
    assert model.generate.call_args.kwargs["max_new_tokens"] == (
        INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_NEW_TOKENS
    )


def test_pre_process_generation_flips_bgr_chw_tensors() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.return_value = {"input_ids": torch.tensor([[1]], dtype=torch.int64)}
    mage_vl = _mage_vl(processor=processor)
    image = torch.zeros((3, 4, 4), dtype=torch.uint8)
    image[0] = 255  # blue channel of a BGR CHW tensor

    mage_vl.pre_process_generation(
        images=image, prompt="what is this?", input_color_format="bgr"
    )

    passed_image = processor.call_args.kwargs["images"]
    assert passed_image[2].max() == 255, "blue must end up in the last channel"
    assert passed_image[0].max() == 0


def test_pre_process_generation_flips_bgr_hwc_tensors() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.return_value = {"input_ids": torch.tensor([[1]], dtype=torch.int64)}
    mage_vl = _mage_vl(processor=processor)
    image = torch.zeros((4, 4, 3), dtype=torch.uint8)
    image[..., 0] = 255

    mage_vl.pre_process_generation(
        images=image, prompt="what is this?", input_color_format="bgr"
    )

    passed_image = processor.call_args.kwargs["images"]
    assert passed_image[..., 2].max() == 255
    assert passed_image[..., 0].max() == 0


def test_pre_process_generation_leaves_undeclared_tensors_alone() -> None:
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.return_value = {"input_ids": torch.tensor([[1]], dtype=torch.int64)}
    mage_vl = _mage_vl(processor=processor)
    image = torch.zeros((3, 4, 4), dtype=torch.uint8)
    image[0] = 255

    mage_vl.pre_process_generation(images=image, prompt="what is this?")

    passed_image = processor.call_args.kwargs["images"]
    assert passed_image[0].max() == 255, "tensors without a declared format pass through"


def test_video_pre_processing_wraps_missing_dependency_errors(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.side_effect = ModuleNotFoundError("No module named 'codec_video_prep'")
    mage_vl = _mage_vl(processor=processor)

    with pytest.raises(MissingDependencyError) as error:
        mage_vl.pre_process_generation(video=str(video))

    assert "codec-video-prep" in str(error.value)


def test_video_pre_processing_wraps_missing_binary_errors(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.side_effect = FileNotFoundError("No such file or directory: 'ffprobe'")
    mage_vl = _mage_vl(processor=processor)

    with pytest.raises(MissingDependencyError):
        mage_vl.pre_process_generation(video=str(video))


def test_video_pre_processing_wraps_unexpected_errors(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    processor = MagicMock()
    processor.apply_chat_template.return_value = "prompt"
    processor.side_effect = ValueError("bad canvas")
    mage_vl = _mage_vl(processor=processor)

    from inference_models.errors import ModelRuntimeError

    with pytest.raises(ModelRuntimeError):
        mage_vl.pre_process_generation(video=str(video))


def test_attn_implementation_is_sdpa_on_cpu() -> None:
    assert _get_mage_vl_attn_implementation(torch.device("cpu")) == "sdpa"


def test_attn_implementation_is_sdpa_without_flash_attn() -> None:
    device = torch.device("cuda", 0)
    from unittest.mock import patch

    with patch(
        "inference_models.models.mage_vl.mage_vl_hf.is_flash_attn_2_available",
        return_value=False,
    ):
        assert _get_mage_vl_attn_implementation(device) == "sdpa"


def test_post_process_generation_strips_decoded_text() -> None:
    processor = MagicMock()
    processor.tokenizer.batch_decode.return_value = ["  a dog  \n"]
    mage_vl = _mage_vl(processor=processor)

    assert mage_vl.post_process_generation(
        generated_ids=torch.tensor([[1, 2]])
    ) == ["a dog"]
