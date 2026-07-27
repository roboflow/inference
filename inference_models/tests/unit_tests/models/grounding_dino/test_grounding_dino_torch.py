from unittest.mock import MagicMock, call, patch

import pytest
import torch
from huggingface_hub.errors import LocalEntryNotFoundError

from inference_models.models.grounding_dino import grounding_dino_torch


def test_grounding_dino_resolves_missing_text_encoder_to_dependency_cache(
    tmp_path,
) -> None:
    package_dir = tmp_path / "grounding-dino"
    package_dir.mkdir()
    dependency_cache = tmp_path / "hf_home" / "hub"
    text_encoder_snapshot = dependency_cache / "bert-snapshot"
    loaded_model = MagicMock()
    loaded_model.to.return_value = loaded_model

    with (
        patch.object(
            grounding_dino_torch,
            "HF_HUB_CACHE",
            str(dependency_cache),
        ),
        patch.object(grounding_dino_torch, "OFFLINE_MODE", True),
        patch.object(
            grounding_dino_torch,
            "get_model_package_contents",
            return_value={
                "weights.pth": str(package_dir / "weights.pth"),
                "config.py": str(package_dir / "config.py"),
            },
        ),
        patch.object(
            grounding_dino_torch,
            "snapshot_download",
            return_value=str(text_encoder_snapshot),
        ) as snapshot_download,
        patch.object(
            grounding_dino_torch,
            "load_model",
            return_value=loaded_model,
        ) as load_model,
    ):
        model = (
            grounding_dino_torch.GroundingDinoForObjectDetectionTorch.from_pretrained(
                model_name_or_path=str(package_dir),
                device=torch.device("cpu"),
            )
        )

    snapshot_download.assert_called_once_with(
        repo_id="google-bert/bert-base-uncased",
        cache_dir=str(dependency_cache),
        local_files_only=True,
        allow_patterns=grounding_dino_torch.BERT_SNAPSHOT_ALLOW_PATTERNS,
    )
    assert load_model.call_args.kwargs["text_encoder_type"] == str(
        text_encoder_snapshot
    )
    assert model._model is loaded_model


@pytest.mark.parametrize("offline_mode", [False, True])
def test_grounding_dino_uses_legacy_bert_cache_as_fallback(
    tmp_path,
    offline_mode,
) -> None:
    dependency_cache = tmp_path / "hf_home" / "hub"
    legacy_snapshot = dependency_cache / "legacy-bert-snapshot"

    with (
        patch.object(
            grounding_dino_torch,
            "HF_HUB_CACHE",
            str(dependency_cache),
        ),
        patch.object(grounding_dino_torch, "OFFLINE_MODE", offline_mode),
        patch.object(
            grounding_dino_torch,
            "snapshot_download",
            side_effect=[
                LocalEntryNotFoundError("canonical snapshot is not cached"),
                str(legacy_snapshot),
            ],
        ) as snapshot_download,
    ):
        result = grounding_dino_torch._download_bert_snapshot()

    canonical_kwargs = {
        "cache_dir": str(dependency_cache),
        "local_files_only": offline_mode,
        "allow_patterns": grounding_dino_torch.BERT_SNAPSHOT_ALLOW_PATTERNS,
    }
    legacy_kwargs = {**canonical_kwargs, "local_files_only": True}
    assert result == str(legacy_snapshot)
    assert snapshot_download.call_args_list == [
        call(repo_id="google-bert/bert-base-uncased", **canonical_kwargs),
        call(repo_id="bert-base-uncased", **legacy_kwargs),
    ]


def test_grounding_dino_prefers_packaged_text_encoder(tmp_path) -> None:
    package_dir = tmp_path / "grounding-dino"
    text_encoder_dir = package_dir / "text_encoder"
    text_encoder_dir.mkdir(parents=True)
    loaded_model = MagicMock()
    loaded_model.to.return_value = loaded_model

    with (
        patch.object(
            grounding_dino_torch,
            "get_model_package_contents",
            return_value={
                "weights.pth": str(package_dir / "weights.pth"),
                "config.py": str(package_dir / "config.py"),
            },
        ),
        patch.object(
            grounding_dino_torch,
            "snapshot_download",
        ) as snapshot_download,
        patch.object(
            grounding_dino_torch,
            "load_model",
            return_value=loaded_model,
        ) as load_model,
    ):
        grounding_dino_torch.GroundingDinoForObjectDetectionTorch.from_pretrained(
            model_name_or_path=str(package_dir),
            device=torch.device("cpu"),
        )

    snapshot_download.assert_not_called()
    assert load_model.call_args.kwargs["text_encoder_type"] == str(text_encoder_dir)
