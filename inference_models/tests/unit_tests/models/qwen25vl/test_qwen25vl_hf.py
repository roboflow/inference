import hashlib
import json

from inference_models.models.qwen25vl import qwen25vl_hf


def test_patch_preprocessor_config_does_not_mutate_cached_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    blob_path = tmp_path / "shared-preprocessor-blob"
    blob_path.write_text(
        json.dumps(
            {
                "image_processor_type": "Qwen2_5_VLImageProcessor",
                "size": {"longest_edge": 1280},
            },
            indent=2,
        )
    )
    config_path = tmp_path / "preprocessor_config.json"
    config_path.symlink_to(blob_path)
    original_content = blob_path.read_bytes()
    original_md5 = hashlib.md5(original_content).hexdigest()
    monkeypatch.delattr(
        qwen25vl_hf.Qwen2_5_VLProcessor,
        "image_processor_class",
        raising=False,
    )

    qwen25vl_hf._patch_preprocessor_config(cache_dir=str(tmp_path))

    assert config_path.is_symlink()
    assert blob_path.read_bytes() == original_content
    assert hashlib.md5(blob_path.read_bytes()).hexdigest() == original_md5
    assert (
        qwen25vl_hf.Qwen2_5_VLProcessor.image_processor_class == "Qwen2VLImageProcessor"
    )
