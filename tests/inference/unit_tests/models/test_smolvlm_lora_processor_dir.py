from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from inference.models.smolvlm import smolvlm

RESOLVED_BASE_DIR = "/model-cache/~lora-bases-smolvlm2-main-0123456789"


@pytest.mark.parametrize("base_model_id", ["smolvlm2", "smolvlm2/smolvlm-256m"])
def test_lora_smolvlm_loads_processor_from_the_extracted_base_dir(
    monkeypatch, base_model_id
) -> None:
    # given
    monkeypatch.setattr(smolvlm, "load_compatible_adapter_config", lambda **kwargs: {})
    monkeypatch.setattr(
        smolvlm,
        "LoraConfig",
        lambda **kwargs: SimpleNamespace(
            base_model_name_or_path=base_model_id, revision="main"
        ),
    )
    monkeypatch.setattr(smolvlm, "PeftModel", MagicMock())
    monkeypatch.setattr(smolvlm, "remove_extracted_archive_if_online", MagicMock())
    monkeypatch.setattr(smolvlm, "is_flash_attn_2_available", lambda: False)
    model = smolvlm.LoRASmolVLM.__new__(smolvlm.LoRASmolVLM)
    model.cache_dir = "/model-cache/adapter"
    model.get_lora_base_from_roboflow = MagicMock(return_value=RESOLVED_BASE_DIR)
    model.transformers_class = MagicMock()
    model.processor_class = MagicMock()

    # when
    model.initialize_model()

    # then
    model.transformers_class.from_pretrained.assert_called_once()
    assert (
        model.transformers_class.from_pretrained.call_args.args[0] == RESOLVED_BASE_DIR
    )
    model.processor_class.from_pretrained.assert_called_once_with(
        RESOLVED_BASE_DIR, local_files_only=smolvlm.OFFLINE_MODE
    )
