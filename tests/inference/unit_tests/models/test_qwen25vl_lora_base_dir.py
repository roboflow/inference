import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from transformers import Qwen2_5_VLForConditionalGeneration

from inference.models.qwen25vl import qwen25vl

CLASSES = [qwen25vl.Qwen25VL, qwen25vl.LoRAQwen25VL]


@pytest.fixture
def package(tmp_path, monkeypatch):
    model_cache_dir = tmp_path / "model-cache"
    model_cache_dir.mkdir()
    base_dir = model_cache_dir / "~lora-bases-qwen-qwen25vl-7b-main-0123456789"
    base_dir.mkdir()
    (base_dir / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "Qwen2VLImageProcessor"})
    )
    package_dir = model_cache_dir / "adapter"
    package_dir.mkdir()
    monkeypatch.setattr(qwen25vl, "MODEL_CACHE_DIR", str(model_cache_dir))
    monkeypatch.setattr(qwen25vl, "load_compatible_adapter_config", lambda **kwargs: {})
    monkeypatch.setattr(
        qwen25vl,
        "LoraConfig",
        lambda **kwargs: SimpleNamespace(
            base_model_name_or_path="qwen/qwen25vl-7b", revision="main"
        ),
    )
    monkeypatch.setattr(qwen25vl, "PeftModel", MagicMock())
    monkeypatch.setattr(qwen25vl, "remove_extracted_archive_if_online", MagicMock())
    return package_dir, base_dir


def _model(model_class, package_dir, base_dir):
    model = model_class.__new__(model_class)
    model.cache_dir = str(package_dir)
    model.huggingface_token = None
    model.use_quantization = False
    model.dtype = "bfloat16"
    model.get_lora_base_from_roboflow = MagicMock(return_value=str(base_dir))
    model.transformers_class = MagicMock()
    model.processor_class = MagicMock()
    return model


@pytest.mark.parametrize("model_class", CLASSES)
def test_preprocessor_config_is_read_from_the_extracted_base_dir(
    package, model_class
) -> None:
    # given
    package_dir, base_dir = package
    (package_dir / "chat_template.json").write_text(
        json.dumps({"chat_template": "json template"})
    )
    model = _model(model_class, package_dir, base_dir)

    # when
    model.initialize_model()

    # then
    assert model.transformers_class.from_pretrained.call_args.args == (str(base_dir),)
    processor_load = model.processor_class.from_pretrained.call_args
    assert processor_load.args == (str(base_dir),)
    assert processor_load.kwargs["chat_template"] == "json template"


@pytest.mark.parametrize("model_class", CLASSES)
@pytest.mark.parametrize("template_in", ["package", "base"])
def test_chat_template_jinja_is_accepted(package, model_class, template_in) -> None:
    # given
    package_dir, base_dir = package
    directory = package_dir if template_in == "package" else base_dir
    (directory / "chat_template.jinja").write_text("jinja template")
    model = _model(model_class, package_dir, base_dir)

    # when
    model.initialize_model()

    # then
    processor_load = model.processor_class.from_pretrained.call_args
    assert processor_load.kwargs["chat_template"] == "jinja template"


@pytest.mark.parametrize("model_class", CLASSES)
def test_missing_chat_template_fails_the_load(package, model_class) -> None:
    # given
    package_dir, base_dir = package
    model = _model(model_class, package_dir, base_dir)

    # when / then
    with pytest.raises(FileNotFoundError, match="chat_template"):
        model.initialize_model()


@pytest.mark.parametrize("model_class", CLASSES)
def test_qwen25vl_loads_through_its_conditional_generation_class(
    model_class,
) -> None:
    assert model_class.transformers_class is Qwen2_5_VLForConditionalGeneration
