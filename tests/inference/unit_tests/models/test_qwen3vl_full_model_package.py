from unittest.mock import MagicMock

import pytest
import torch

from inference.models.qwen3vl import qwen3vl


class AdapterPathTaken(Exception):
    pass


def _model(tmp_path) -> qwen3vl.Qwen3VL:
    model = qwen3vl.Qwen3VL.__new__(qwen3vl.Qwen3VL)
    model.cache_dir = str(tmp_path)
    model.dtype = torch.bfloat16
    model.transformers_class = MagicMock()
    model.processor_class = MagicMock()
    return model


def _adapter_loader(**kwargs):
    raise AdapterPathTaken()


def test_full_fine_tune_package_loads_from_the_package_dir(
    tmp_path, monkeypatch
) -> None:
    # given
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(qwen3vl, "load_compatible_adapter_config", _adapter_loader)
    model = _model(tmp_path)

    # when
    model.initialize_model()

    # then
    load = model.transformers_class.from_pretrained
    assert load.call_args.args == (str(tmp_path),)
    assert load.call_args.kwargs["torch_dtype"] == torch.bfloat16
    assert load.call_args.kwargs["local_files_only"] is True
    model.processor_class.from_pretrained.assert_called_once_with(
        str(tmp_path),
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        local_files_only=True,
    )
    assert model.model is load.return_value.eval.return_value.to.return_value


def test_adapter_package_keeps_the_adapter_path(tmp_path, monkeypatch) -> None:
    # given
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(qwen3vl, "load_compatible_adapter_config", _adapter_loader)
    model = _model(tmp_path)

    # when / then
    with pytest.raises(AdapterPathTaken):
        model.initialize_model()
    model.transformers_class.from_pretrained.assert_not_called()
