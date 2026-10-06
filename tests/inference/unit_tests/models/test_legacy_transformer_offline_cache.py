import json


def test_adapter_config_is_sanitized_in_memory_without_offline_write(
    monkeypatch,
    tmp_path,
) -> None:
    from inference.models.transformers import transformers

    config_path = tmp_path / "adapter_config.json"
    original_config = {
        "base_model_name_or_path": "qwen/base",
        "r": 8,
        "eva_config": {"rho": 2.0},
        "lora_bias": True,
    }
    config_path.write_text(json.dumps(original_config))
    monkeypatch.setattr(transformers, "OFFLINE_MODE", True)

    sanitized_config = transformers.load_compatible_adapter_config(
        config_file=str(config_path),
        unsupported_keys=["eva_config", "lora_bias"],
    )

    assert sanitized_config == {
        "base_model_name_or_path": "qwen/base",
        "r": 8,
    }
    assert json.loads(config_path.read_text()) == original_config


def test_adapter_config_is_persisted_during_online_warm(
    monkeypatch,
    tmp_path,
) -> None:
    from inference.models.transformers import transformers

    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_model_name_or_path": "qwen/base",
                "r": 8,
                "exclude_modules": ["vision"],
            }
        )
    )
    monkeypatch.setattr(transformers, "OFFLINE_MODE", False)

    sanitized_config = transformers.load_compatible_adapter_config(
        config_file=str(config_path),
        unsupported_keys=["exclude_modules"],
    )

    assert json.loads(config_path.read_text()) == sanitized_config
    assert "exclude_modules" not in sanitized_config


def test_extracted_archives_are_never_removed_offline(
    monkeypatch,
    tmp_path,
) -> None:
    from inference.models.transformers import transformers

    archive_path = tmp_path / "weights.tar.gz"
    archive_path.write_bytes(b"cached archive")
    monkeypatch.setattr(transformers, "OFFLINE_MODE", True)

    transformers.remove_extracted_archive_if_online(str(archive_path))

    assert archive_path.read_bytes() == b"cached archive"


def test_extracted_archives_keep_online_cleanup_behavior(
    monkeypatch,
    tmp_path,
) -> None:
    from inference.models.transformers import transformers

    archive_path = tmp_path / "weights.tar.gz"
    archive_path.write_bytes(b"cached archive")
    monkeypatch.setattr(transformers, "OFFLINE_MODE", False)

    transformers.remove_extracted_archive_if_online(str(archive_path))

    assert not archive_path.exists()


def test_qwen_preprocessor_config_is_read_only_offline(
    monkeypatch,
    tmp_path,
) -> None:
    from inference.models.qwen25vl import qwen25vl

    config_path = tmp_path / "preprocessor_config.json"
    compatible_config = {
        "image_processor_type": "Qwen2VLImageProcessor",
        "size": 512,
    }
    config_path.write_text(json.dumps(compatible_config))
    monkeypatch.setattr(qwen25vl, "OFFLINE_MODE", True)

    qwen25vl._patch_preprocessor_config(str(tmp_path))

    assert json.loads(config_path.read_text()) == compatible_config


def test_qwen_incompatible_preprocessor_requires_online_rewarm(
    monkeypatch,
    tmp_path,
) -> None:
    import pytest

    from inference.models.qwen25vl import qwen25vl

    config_path = tmp_path / "preprocessor_config.json"
    incompatible_config = {
        "image_processor_type": "Qwen2_5_VLImageProcessor",
        "size": 512,
    }
    config_path.write_text(json.dumps(incompatible_config))
    monkeypatch.setattr(qwen25vl, "OFFLINE_MODE", True)

    with pytest.raises(ValueError, match="Re-warm"):
        qwen25vl._patch_preprocessor_config(str(tmp_path))

    assert json.loads(config_path.read_text()) == incompatible_config


def test_qwen_required_file_lists_cover_constructor_inputs() -> None:
    from inference.models.qwen3vl.qwen3vl import Qwen3VL
    from inference.models.qwen25vl.qwen25vl import LoRAQwen25VL, Qwen25VL

    qwen25_files = Qwen25VL.get_infer_bucket_file_list(object.__new__(Qwen25VL))
    lora_qwen25_files = LoRAQwen25VL.get_infer_bucket_file_list(
        object.__new__(LoRAQwen25VL)
    )
    qwen3_files = Qwen3VL.get_infer_bucket_file_list(object.__new__(Qwen3VL))

    assert {"adapter_config.json", "chat_template.json"} <= set(qwen25_files)
    assert {"adapter_config.json", "chat_template.json"} <= set(lora_qwen25_files)
    assert "adapter_config.json" in qwen3_files
