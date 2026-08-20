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
    verified = core.AutoModel.verify_offline_models(model_id=MODEL_ID)
    purged = core.AutoModel.purge_offline_model(model_id="my-alias")

    # then
    assert len(listed) == 1
    assert listed[0]["canonical_model_id"] == MODEL_ID
    assert listed[0]["requested_aliases"] == ["my-alias"]
    assert listed[0]["packages"][0]["presence"] == "missing"
    assert {result["status"] for result in verified} == {"missing"}
    assert purged is True
    assert core.AutoModel.list_offline_models() == []


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
