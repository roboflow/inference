import json
from pathlib import Path

import numpy as np
import pytest
import torch

from inference_models import ActionRecognitionContract, get_action_recognition_contract
from inference_models.errors import (
    CorruptedModelPackageError,
    ModelImplementationNotFoundError,
    ModelRetrievalError,
)
from inference_models.models.auto_loaders import action_recognition
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.base.action_recognition import (
    SLIDING_WINDOW_MODE,
    WHOLE_VIDEO_MODE,
    ActionRecognitionModel,
    VideoSampling,
)
from inference_models.models.cosmos3.cosmos3_action_recognition import (
    Cosmos3EdgeActionRecognition,
)
from inference_models.models.vjepa2_1.model import VJepaActionRecognition
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    LocalFileArtefactSpecs,
    ModelMetadata,
    ModelPackageMetadata,
)

from .test_vjepa2_1_model import config as vjepa_config

COSMOS_CONFIG = {
    "video_pre_processing": {
        "mode": "sliding_window",
        "sample_fps": 12.0,
        "window_frames": 32,
        "window_seconds": 32 / 12,
        "min_frames": 4,
        "max_frame_side": 360,
    }
}


def _write_vjepa_package(package_dir: Path) -> dict:
    config = vjepa_config()
    (package_dir / "inference_config.json").write_text(json.dumps(config))
    (package_dir / "class_names.txt").write_text("a\nb\n")

    return config


def test_vjepa_contract_matches_the_model_it_describes(tmp_path) -> None:
    config = _write_vjepa_package(tmp_path)
    loaded = VJepaActionRecognition(None, config, ["a", "b"], torch.device("cpu"))

    contract = VJepaActionRecognition.contract_from_package(str(tmp_path))

    assert contract.class_names == loaded.class_names
    assert contract.video_sampling == loaded.video_sampling
    assert (
        contract.supports_confidence,
        contract.supports_observed_duration,
        contract.end_exclusive_spans,
    ) == (True, True, True)
    frame = np.random.default_rng(0).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        contract.frame_storage_transform(frame), loaded.frame_storage_transform(frame)
    )


def test_vjepa_contract_rejects_what_from_pretrained_rejects(tmp_path) -> None:
    _write_vjepa_package(tmp_path)
    (tmp_path / "class_names.txt").write_text("b\na\n")

    with pytest.raises(ValueError, match="inconsistent"):
        VJepaActionRecognition.contract_from_package(str(tmp_path))


def test_vjepa_contract_needs_its_files(tmp_path) -> None:
    with pytest.raises(CorruptedModelPackageError):
        VJepaActionRecognition.contract_from_package(str(tmp_path))


def test_cosmos_fine_tune_contract_reads_classes_and_recorded_sampling(
    tmp_path,
) -> None:
    (tmp_path / "inference_config.json").write_text(json.dumps(COSMOS_CONFIG))
    (tmp_path / "class_names.txt").write_text("punch\nkick\n")

    contract = Cosmos3EdgeActionRecognition.contract_from_package(str(tmp_path))

    assert contract == ActionRecognitionContract(
        class_names=["punch", "kick"],
        video_sampling=VideoSampling(
            window_seconds=32 / 12,
            sample_fps=12.0,
            min_frames=4,
            max_frame_side=360,
            mode=SLIDING_WINDOW_MODE,
            max_frames=32,
        ),
    )


def test_cosmos_zero_shot_contract_reads_whole_videos(tmp_path) -> None:
    # Without a class list the model is zero-shot, which a loaded model
    # serves in whole-video mode whatever the package records.
    (tmp_path / "inference_config.json").write_text(json.dumps(COSMOS_CONFIG))

    contract = Cosmos3EdgeActionRecognition.contract_from_package(str(tmp_path))

    assert contract.class_names is None
    assert contract.video_sampling.mode == WHOLE_VIDEO_MODE


def test_cosmos_contract_rejects_a_class_the_span_format_cannot_express(
    tmp_path,
) -> None:
    (tmp_path / "class_names.txt").write_text("left|right\n")

    with pytest.raises(CorruptedModelPackageError, match="'\\|'"):
        Cosmos3EdgeActionRecognition.contract_from_package(str(tmp_path))


def test_a_model_without_a_contract_reader_says_so(tmp_path) -> None:
    class _Opaque(ActionRecognitionModel):
        pass

    with pytest.raises(ModelImplementationNotFoundError, match="_Opaque"):
        _Opaque.contract_from_package(str(tmp_path))


def _metadata(architecture: str, backend: BackendType, artefacts) -> ModelMetadata:
    return ModelMetadata(
        model_id="ws/actions/1",
        model_architecture=architecture,
        model_packages=[
            ModelPackageMetadata(
                package_id="pkg-1", backend=backend, package_artefacts=artefacts
            )
        ],
    )


def _remote_artefacts(*names: str):
    return [
        FileDownloadSpecs(download_url=f"https://repo/{name}", file_handle=name)
        for name in names
    ]


def _fake_platform(monkeypatch, metadata: ModelMetadata, files: dict) -> list:
    requested = []

    def get_model_from_provider(model_id, provider, api_key):
        requested.append(("metadata", model_id, provider, api_key))
        return metadata

    def download_files_to_directory(target_dir, files_specs, **kwargs):
        for handle, url, _ in files_specs:
            requested.append(("download", handle, url))
            (Path(target_dir) / handle).write_text(files[handle])

    monkeypatch.setattr(
        action_recognition, "get_model_from_provider", get_model_from_provider
    )
    monkeypatch.setattr(
        action_recognition, "download_files_to_directory", download_files_to_directory
    )

    return requested


def test_contract_downloads_only_the_files_the_model_class_reads(monkeypatch) -> None:
    files = {
        "inference_config.json": json.dumps(vjepa_config()),
        "class_names.txt": "a\nb\n",
    }
    metadata = _metadata(
        "vjepa2-1-vitb-384",
        BackendType.TORCH,
        _remote_artefacts("model.safetensors", *files),
    )
    requested = _fake_platform(monkeypatch, metadata=metadata, files=files)

    contract = get_action_recognition_contract("ws/actions/1", api_key="key")

    assert requested == [
        ("metadata", "ws/actions/1", "roboflow", "key"),
        ("download", "inference_config.json", "https://repo/inference_config.json"),
        ("download", "class_names.txt", "https://repo/class_names.txt"),
    ]
    assert contract.class_names == ["a", "b"]
    assert contract.video_sampling.max_frames == 4
    assert contract.end_exclusive_spans is True


def test_contract_of_a_cosmos_fine_tune_comes_from_its_package(monkeypatch) -> None:
    files = {
        "inference_config.json": json.dumps(COSMOS_CONFIG),
        "class_names.txt": "punch\n",
    }
    metadata = _metadata(
        "cosmos3-edge",
        BackendType.HF,
        _remote_artefacts("model.safetensors", "tokenizer.json", *files),
    )
    _fake_platform(monkeypatch, metadata=metadata, files=files)

    contract = get_action_recognition_contract("ws/actions/1")

    assert contract.class_names == ["punch"]
    assert contract.video_sampling.max_frames == 32
    assert contract.frame_storage_transform is None


def test_an_architecture_without_an_action_recognition_class_is_refused(
    monkeypatch,
) -> None:
    metadata = _metadata("yolov8", BackendType.ONNX, _remote_artefacts("w.onnx"))
    _fake_platform(monkeypatch, metadata=metadata, files={})

    with pytest.raises(ModelImplementationNotFoundError, match="'yolov8'"):
        get_action_recognition_contract("ws/actions/1")


def test_a_contract_file_that_cannot_be_downloaded_is_an_error(monkeypatch) -> None:
    # Reading defaults instead would serve a contract the package overrides.
    metadata = _metadata(
        "cosmos3-edge",
        BackendType.HF,
        [LocalFileArtefactSpecs(file_handle="inference_config.json", md5_hash="x")],
    )
    _fake_platform(monkeypatch, metadata=metadata, files={})

    with pytest.raises(ModelRetrievalError, match="inference_config.json"):
        get_action_recognition_contract("ws/actions/1")
