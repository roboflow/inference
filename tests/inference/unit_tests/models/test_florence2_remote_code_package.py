import json
from unittest.mock import MagicMock

import pytest

from inference.models.florence2 import florence2
from inference.models.transformers import TransformerModel


def _package(tmp_path, text_model_type: str, with_modeling_file: bool = True):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"model_type": text_model_type}})
    )
    if with_modeling_file:
        (tmp_path / "modeling_florence2.py").write_text("")
    model = florence2.Florence2.__new__(florence2.Florence2)
    model.cache_dir = str(tmp_path)
    model.dtype = "float32"
    return model


@pytest.fixture
def loaders(monkeypatch):
    native_loader = MagicMock()
    monkeypatch.setattr(TransformerModel, "initialize_model", native_loader)
    bundled = {
        "Florence2ForConditionalGeneration": MagicMock(),
        "Florence2Processor": MagicMock(),
    }
    monkeypatch.setattr(
        florence2,
        "import_class_from_file",
        lambda file_path, class_name: bundled[class_name],
    )
    return native_loader, bundled


def test_remote_code_package_loads_through_its_bundled_classes(
    tmp_path, loaders
) -> None:
    # given
    native_loader, bundled = loaders
    model = _package(tmp_path, "florence2_language")
    model_class = bundled["Florence2ForConditionalGeneration"]
    processor_class = bundled["Florence2Processor"]
    config = model_class.config_class.from_pretrained.return_value

    # when
    model.initialize_model()

    # then
    native_loader.assert_not_called()
    model_class.config_class.from_pretrained.assert_called_once_with(str(tmp_path))
    assert model_class.from_pretrained.call_args.args == (str(tmp_path),)
    assert model_class.from_pretrained.call_args.kwargs["config"] is config
    processor_class.from_pretrained.assert_called_once_with(
        str(tmp_path), config=config, local_files_only=True
    )
    assert model.processor is processor_class.from_pretrained.return_value


@pytest.mark.parametrize(
    "text_model_type, with_modeling_file",
    [("florence2", True), ("florence2_language", False)],
)
def test_other_packages_keep_the_native_loader(
    tmp_path, loaders, text_model_type, with_modeling_file
) -> None:
    # given
    native_loader, bundled = loaders
    model = _package(tmp_path, text_model_type, with_modeling_file)

    # when
    model.initialize_model()

    # then
    native_loader.assert_called_once()
    bundled[
        "Florence2ForConditionalGeneration"
    ].config_class.from_pretrained.assert_not_called()
