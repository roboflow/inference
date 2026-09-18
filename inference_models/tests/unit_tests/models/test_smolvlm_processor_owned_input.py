import copy
import json

import pytest

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.smolvlm.inference_config import parse_smolvlm_inference_config


NATIVE_NO_RESIZE = {
    "image_pre_processing": None,
    "network_input": {
        "training_input_size": None,
        "dynamic_spatial_size_supported": True,
        "dynamic_spatial_size_mode": {"type": "any-size"},
        "color_mode": "rgb",
        "resize_mode": "stretch",
        "padding_value": None,
        "input_channels": 3,
        "scaling_factor": None,
        "normalization": None,
    },
}


def parse(tmp_path, value):
    path = tmp_path / "inference_config.json"
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    result = parse_smolvlm_inference_config(str(path))
    assert path.read_bytes() == before
    return result


def test_native_no_resize_delegates_to_packaged_processor(tmp_path):
    assert parse(tmp_path, NATIVE_NO_RESIZE) is None


def test_fixed_training_resize_preserves_existing_parser(tmp_path):
    config = copy.deepcopy(NATIVE_NO_RESIZE)
    config["network_input"]["training_input_size"] = {"width": 640, "height": 480}
    result = parse(tmp_path, config)
    assert result.network_input.training_input_size.width == 640
    assert result.network_input.training_input_size.height == 480


@pytest.mark.parametrize("key,value", [
    ("dynamic_spatial_size_supported", False),
    ("dynamic_spatial_size_mode", {"type": "pad-to-be-divisible", "value": 32}),
    ("color_mode", "bgr"),
    ("scaling_factor", 0.003921568627),
    ("normalization", [[0.5], [0.5]]),
    ("input_channels", 1),
    ("unknown_transform", True),
])
def test_no_resize_never_discards_nonidentity_transforms(tmp_path, key, value):
    config = copy.deepcopy(NATIVE_NO_RESIZE)
    config["network_input"][key] = value
    with pytest.raises(CorruptedModelPackageError): parse(tmp_path, config)


def test_no_resize_does_not_discard_image_preprocessing(tmp_path):
    config = copy.deepcopy(NATIVE_NO_RESIZE)
    config["image_pre_processing"] = {"grayscale": {"enabled": True}}
    with pytest.raises(CorruptedModelPackageError): parse(tmp_path, config)


def test_missing_training_size_preserves_shared_any_size_parser(tmp_path):
    config = copy.deepcopy(NATIVE_NO_RESIZE)
    del config["network_input"]["training_input_size"]
    result = parse(tmp_path, config)
    # The shared schema permits omitted sizes for dynamic any-size inputs.
    # Only explicit null enters the Smol-specific processor-owned shortcut;
    # omission must retain the existing parsed preprocessing configuration.
    assert result is not None
    assert result.network_input.training_input_size is None
    assert result.network_input.dynamic_spatial_size_supported is True
    assert result.network_input.dynamic_spatial_size_mode.type == "any-size"


def test_missing_training_size_still_rejects_static_inputs(tmp_path):
    config = copy.deepcopy(NATIVE_NO_RESIZE)
    del config["network_input"]["training_input_size"]
    config["network_input"]["dynamic_spatial_size_supported"] = False
    with pytest.raises(CorruptedModelPackageError):
        parse(tmp_path, config)


def test_missing_training_size_preserves_shared_normalization(tmp_path):
    config = copy.deepcopy(NATIVE_NO_RESIZE)
    del config["network_input"]["training_input_size"]
    config["network_input"]["normalization"] = [[0.5] * 3, [0.25] * 3]
    result = parse(tmp_path, config)
    assert result.network_input.normalization == ([0.5] * 3, [0.25] * 3)


def test_processor_receives_original_variable_image_dimensions(tmp_path):
    import numpy as np
    import torch
    from inference_models.models.smolvlm.smolvlm_hf import SmolVLMHF

    class Processor:
        def apply_chat_template(self, conversations, **kwargs): return ["prompt"]
        def __call__(self, **kwargs):
            self.images = kwargs["images"]
            return torch.zeros(1)

    model = object.__new__(SmolVLMHF)
    model._inference_config = parse(tmp_path, NATIVE_NO_RESIZE)
    model._processor = Processor()
    model._device = torch.device("cpu")
    model._torch_dtype = torch.float32
    images = [np.zeros((17, 23, 3), dtype=np.uint8), np.zeros((31, 19, 3), dtype=np.uint8)]
    model.pre_process_generation(images, "Describe this image", input_color_format="rgb")
    assert [tuple(image.shape) for image in model._processor.images] == [(3, 17, 23), (3, 31, 19)]
