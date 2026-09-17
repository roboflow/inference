"""End-to-end unit tests for OFFLINE_MODE_WARM_UP recording and OFFLINE serving."""

from datetime import datetime
from unittest import mock

import pytest

from inference_models.errors import ModelRetrievalError
from inference_models.models.auto_loaders import core
from inference_models.models.auto_loaders.auto_resolution_cache import (
    AutoResolutionCacheEntry,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers import core as weights_core
from inference_models.weights_providers import offline_registry
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    ModelMetadata,
    ModelPackageMetadata,
    ONNXPackageDetails,
    Quantization,
)

MODEL_ID = "workspace/model/1"


def _provider_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id=MODEL_ID,
        model_architecture="rfdetr",
        task_type="object-detection",
        model_packages=[
            ModelPackageMetadata(
                package_id="pkgonnx",
                backend=BackendType.ONNX,
                package_artefacts=[
                    FileDownloadSpecs(
                        download_url="https://signed.example/weights.onnx",
                        file_handle="weights.onnx",
                        md5_hash="a" * 32,
                    ),
                ],
                quantization=Quantization.FP32,
                onnx_package_details=ONNXPackageDetails(opset=19),
                trusted_source=True,
            ),
        ],
    )


class _StubEntryCache:
    def __init__(self, entry) -> None:
        self._entry = entry

    def retrieve(self, auto_negotiation_hash):
        return self._entry

    def register(self, auto_negotiation_hash, cache_entry) -> None:
        pass

    def invalidate(self, auto_negotiation_hash) -> None:
        pass


def _cache_entry() -> AutoResolutionCacheEntry:
    return AutoResolutionCacheEntry(
        model_id=MODEL_ID,
        model_package_id="pkgonnx",
        resolved_files=[],
        model_architecture="rfdetr",
        task_type="object-detection",
        backend_type=BackendType.ONNX,
        created_at=datetime.now(),
    )


@pytest.fixture
def registry_home(tmp_path):
    with mock.patch.object(offline_registry, "INFERENCE_HOME", str(tmp_path)):
        yield tmp_path


def test_exclusive_flags_fail_on_load() -> None:
    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
        core, "OFFLINE_MODE_WARM_UP", True
    ):
        with pytest.raises(ModelRetrievalError) as error:
            core.AutoModel.from_pretrained(MODEL_ID)
    assert "mutually" in str(error.value)


def test_offline_mode_swaps_provider_and_fails_actionably_without_record(
    registry_home,
) -> None:
    # given: OFFLINE_MODE with an empty registry; the network provider must
    # never be contacted
    def _forbidden_provider(*args, **kwargs):
        raise AssertionError("network provider contacted in OFFLINE_MODE")

    # when / then
    with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.dict(
        weights_core.WEIGHTS_PROVIDERS, {"roboflow": _forbidden_provider}
    ):
        with pytest.raises(ModelRetrievalError) as error:
            core.AutoModel.from_pretrained(MODEL_ID)
    assert "OFFLINE_MODE_WARM_UP" in str(error.value)


def test_warm_up_records_on_cache_hit(registry_home) -> None:
    # given: pre-fetch succeeds and the auto-load entry cache serves the model
    sentinel_model = object()

    with mock.patch.object(core, "OFFLINE_MODE_WARM_UP", True), mock.patch.object(
        core,
        "get_model_from_provider",
        return_value=_provider_metadata(),
    ) as provider_mock, mock.patch.object(
        core,
        "attempt_loading_model_with_auto_load_cache",
        return_value=sentinel_model,
    ):
        # when
        result = core.AutoModel.from_pretrained(
            MODEL_ID,
            auto_resolution_cache=_StubEntryCache(entry=_cache_entry()),
        )

    # then
    assert result is sentinel_model
    assert provider_mock.call_count == 1
    record = offline_registry.load_record_raw(model_id=MODEL_ID)
    assert record is not None
    assert record["source"] == "warmup"
    assert set(record["proven"].keys()) == {"pkgonnx"}


def test_warm_up_prefetch_failure_serves_model_without_registration(
    registry_home,
) -> None:
    # given: pre-fetch raises; the cached load still succeeds
    sentinel_model = object()

    with mock.patch.object(core, "OFFLINE_MODE_WARM_UP", True), mock.patch.object(
        core,
        "get_model_from_provider",
        side_effect=ConnectionError("api down"),
    ), mock.patch.object(
        core,
        "attempt_loading_model_with_auto_load_cache",
        return_value=sentinel_model,
    ):
        # when
        result = core.AutoModel.from_pretrained(
            MODEL_ID,
            auto_resolution_cache=_StubEntryCache(entry=_cache_entry()),
        )

    # then
    assert result is sentinel_model
    assert offline_registry.load_record_raw(model_id=MODEL_ID) is None


def test_offline_load_of_online_warmed_package_succeeds(tmp_path) -> None:
    """Regression: warm-up materializes artifacts as shared-blob SYMLINKS and
    writes a v4 manifest with shared_blob storage; the offline path rebuilds
    the package from the registry as package_file declarations. The read-only
    offline leg must not run the mutation guard / identity materialization
    that trips on that storage-classification difference.

    Uses a TORCH package: torch ships in every test environment, so real
    auto-negotiation accepts it without backend-availability stubbing (the
    ONNX backend is absent on bare CI runners)."""
    import hashlib
    import json
    import os

    from inference_models.models.auto_loaders import model_cache_paths

    model_id = MODEL_ID
    weights_content = b"torch weights bytes"
    weights_md5 = hashlib.md5(weights_content).hexdigest()
    torch_metadata = ModelMetadata(
        model_id=model_id,
        model_architecture="rfdetr",
        task_type="object-detection",
        model_packages=[
            ModelPackageMetadata(
                package_id="pkgtorch",
                backend=BackendType.TORCH,
                package_artefacts=[
                    FileDownloadSpecs(
                        download_url="https://signed.example/weights.pt",
                        file_handle="weights.pt",
                        md5_hash=weights_md5,
                    ),
                ],
                quantization=Quantization.FP32,
                trusted_source=True,
            ),
        ],
    )

    with mock.patch.object(
        model_cache_paths, "INFERENCE_HOME", str(tmp_path)
    ), mock.patch.object(offline_registry, "INFERENCE_HOME", str(tmp_path)):
        # materialize the package exactly as the ONLINE warm does:
        # shared blob + symlink into the package dir + v4 manifest
        blob_dir = model_cache_paths.generate_shared_blobs_path()
        os.makedirs(blob_dir)
        blob_path = os.path.join(blob_dir, weights_md5)
        with open(blob_path, "wb") as blob_file:
            blob_file.write(weights_content)
        package_dir = model_cache_paths.generate_model_package_cache_path(
            model_id=model_id, package_id="pkgtorch"
        )
        os.makedirs(package_dir)
        os.symlink(blob_path, os.path.join(package_dir, "weights.pt"))
        with open(os.path.join(package_dir, "model_config.json"), "w") as manifest_file:
            json.dump(
                {
                    "offline_manifest_version": 4,
                    "model_id": model_id,
                    "canonical_model_id": model_id,
                    "model_architecture": "rfdetr",
                    "task_type": "object-detection",
                    "backend_type": "torch",
                    "model_features": None,
                    "trusted_source": True,
                    "model_dependencies": [],
                    "recommended_parameters": None,
                    "quantization": "fp32",
                    "dynamic_batch_size_supported": False,
                    "static_batch_size": 1,
                    "package_artifacts": [
                        {
                            "file_handle": "weights.pt",
                            "md5_hash": weights_md5,
                            "unhashed": False,
                            "sha256_hash": None,
                            "source_hash": None,
                            "storage": "shared_blob",
                        }
                    ],
                    "dependency_package_paths": [],
                },
                manifest_file,
            )
        # warm-up records the provider response in the registry
        offline_registry.record_successful_load(
            model_metadata=torch_metadata,
            requested_model_id=model_id,
            proven_package_id="pkgtorch",
        )

        model_class = mock.MagicMock()
        with mock.patch.object(core, "OFFLINE_MODE", True), mock.patch.object(
            core, "resolve_model_class", return_value=model_class
        ):
            # when: OFFLINE load through the full flow (provider swap,
            # negotiation over recorded metadata, presence-only resolve)
            result = core.AutoModel.from_pretrained(model_id)

    # then
    assert result is model_class.from_pretrained.return_value
    loaded_dir = model_class.from_pretrained.call_args.args[0]
    assert os.path.realpath(loaded_dir) == os.path.realpath(package_dir)


def test_maintenance_classmethods_round_trip(registry_home) -> None:
    # given: one recorded model whose artefacts were never materialized
    metadata = _provider_metadata()
    offline_registry.record_successful_load(
        model_metadata=metadata,
        requested_model_id="my-alias",
        proven_package_id="pkgonnx",
    )

    # when
    listed = core.AutoModel.list_offline_models()
    verified = core.AutoModel.verify_offline_model(model_id=MODEL_ID)
    verified_by_alias = core.AutoModel.verify_offline_model(model_id="my-alias")

    # then
    assert len(listed) == 1
    assert isinstance(listed[0], offline_registry.OfflineModelStatus)
    assert listed[0].canonical_model_id == MODEL_ID
    assert listed[0].requested_aliases == ["my-alias"]
    assert (
        listed[0].packages[0].presence
        is offline_registry.OfflinePackagePresence.MISSING
    )
    # proven flattens to package_id -> last_proven_at datetime
    assert set(listed[0].proven) == {"pkgonnx"}
    assert isinstance(listed[0].proven["pkgonnx"], datetime)
    assert isinstance(listed[0].recorded_at, datetime)
    assert {result.status for result in verified} == {
        offline_registry.OfflineArtefactStatus.MISSING
    }
    assert verified_by_alias == verified  # alias resolves to the same record
    assert core.AutoModel.verify_offline_model(model_id="unknown/model/1") == []


def test_without_warm_up_no_prefetch_happens_on_cache_hit(registry_home) -> None:
    # given
    sentinel_model = object()

    with mock.patch.object(
        core,
        "get_model_from_provider",
        side_effect=AssertionError("provider must not be called on cache hit"),
    ), mock.patch.object(
        core,
        "attempt_loading_model_with_auto_load_cache",
        return_value=sentinel_model,
    ):
        # when
        result = core.AutoModel.from_pretrained(
            MODEL_ID,
            auto_resolution_cache=_StubEntryCache(entry=_cache_entry()),
        )

    # then
    assert result is sentinel_model
    assert offline_registry.load_record_raw(model_id=MODEL_ID) is None
