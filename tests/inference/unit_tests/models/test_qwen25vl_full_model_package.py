import json
from unittest.mock import MagicMock

import pytest

from inference.models.qwen25vl import qwen25vl


class AdapterPathTaken(Exception):
    pass


def _model(tmp_path) -> qwen25vl.Qwen25VL:
    model = qwen25vl.Qwen25VL.__new__(qwen25vl.Qwen25VL)
    model.cache_dir = str(tmp_path)
    model.transformers_class = MagicMock()
    model.processor_class = MagicMock()
    return model


def _adapter_loader(**kwargs):
    raise AdapterPathTaken()


@pytest.mark.parametrize(
    "template_file, template_text, expected",
    [
        ("chat_template.jinja", "jinja template", "jinja template"),
        (
            "chat_template.json",
            json.dumps({"chat_template": "json template"}),
            "json template",
        ),
        (None, None, None),
    ],
)
def test_full_fine_tune_package_loads_from_the_package_dir(
    tmp_path, monkeypatch, template_file, template_text, expected
) -> None:
    # given
    (tmp_path / "config.json").write_text("{}")
    if template_file:
        (tmp_path / template_file).write_text(template_text)
    monkeypatch.setattr(qwen25vl, "load_compatible_adapter_config", _adapter_loader)
    model = _model(tmp_path)

    # when
    model.initialize_model()

    # then
    load = model.transformers_class.from_pretrained
    assert load.call_args.args == (str(tmp_path),)
    assert "quantization_config" not in load.call_args.kwargs
    assert load.call_args.kwargs["local_files_only"] is True
    model.processor_class.from_pretrained.assert_called_once_with(
        str(tmp_path),
        chat_template=expected,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        local_files_only=True,
    )
    assert model.model is load.return_value.eval.return_value


def test_adapter_package_keeps_the_adapter_path(tmp_path, monkeypatch) -> None:
    # given
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(qwen25vl, "load_compatible_adapter_config", _adapter_loader)
    model = _model(tmp_path)

    # when / then
    with pytest.raises(AdapterPathTaken):
        model.initialize_model()
    model.transformers_class.from_pretrained.assert_not_called()
