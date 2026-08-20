import json
import multiprocessing
import os
from unittest import mock

import pytest
from packaging.version import Version

from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers import offline_registry
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    JetsonEnvironmentRequirements,
    LocalFileArtefactSpecs,
    ModelMetadata,
    ModelPackageMetadata,
    PackageSourceType,
    Quantization,
    RecommendedParameters,
    ServerEnvironmentRequirements,
    TRTPackageDetails,
)


def _example_metadata(model_id: str = "workspace/model/1") -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
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
                    FileDownloadSpecs(
                        download_url="https://signed.example/class_names.txt",
                        file_handle="class_names.txt",
                        md5_hash="b" * 32,
                    ),
                ],
                quantization=Quantization.FP32,
                dynamic_batch_size_supported=False,
                static_batch_size=1,
                trusted_source=True,
                environment_requirements=ServerEnvironmentRequirements(
                    cuda_device_cc=Version("8.6"),
                    cuda_device_name="NVIDIA GeForce RTX 3090",
                    driver_version=Version("535.104.05"),
                    cuda_version=Version("12.2"),
                    trt_version=None,
                    os_version="linux",
                ),
            ),
            ModelPackageMetadata(
                package_id="pkgtrt",
                backend=BackendType.TRT,
                package_artefacts=[
                    FileDownloadSpecs(
                        download_url="https://signed.example/engine.plan",
                        file_handle="engine.plan",
                        md5_hash="c" * 32,
                    ),
                ],
                quantization=Quantization.FP16,
                dynamic_batch_size_supported=False,
                static_batch_size=1,
                trusted_source=True,
                trt_package_details=TRTPackageDetails(
                    same_cc_compatible=True,
                    trt_forward_compatible=False,
                ),
                environment_requirements=JetsonEnvironmentRequirements(
                    cuda_device_cc=Version("8.7"),
                    cuda_device_name="Orin",
                    l4t_version=Version("36.3"),
                    jetson_product_name="Jetson AGX Orin",
                    cuda_version=Version("12.2"),
                    trt_version=Version("10.3"),
                    driver_version=None,
                ),
            ),
        ],
        recommended_parameters=RecommendedParameters(confidence=0.4),
    )


@pytest.fixture
def registry_home(tmp_path):
    with mock.patch.object(
        offline_registry, "INFERENCE_HOME", str(tmp_path)
    ):
        yield tmp_path


def test_record_and_load_round_trip(registry_home) -> None:
    # given
    metadata = _example_metadata()

    # when
    registered = offline_registry.record_successful_load(
        model_metadata=metadata,
        requested_model_id="workspace/model/1",
        proven_package_id="pkgonnx",
    )
    loaded = offline_registry.load_model_metadata(model_id="workspace/model/1")

    # then
    assert registered is True
    assert loaded is not None
    assert loaded.model_id == "workspace/model/1"
    assert loaded.model_architecture == "rfdetr"
    assert loaded.task_type == "object-detection"
    assert loaded.recommended_parameters.confidence == 0.4
    assert {p.package_id for p in loaded.model_packages} == {"pkgonnx", "pkgtrt"}
    for package in loaded.model_packages:
        assert package.package_source is PackageSourceType.LOCAL_CACHE
        assert all(
            isinstance(a, LocalFileArtefactSpecs) for a in package.package_artefacts
        )
    onnx_package = next(
        p for p in loaded.model_packages if p.package_id == "pkgonnx"
    )
    assert onnx_package.environment_requirements.cuda_device_cc == Version("8.6")
    trt_package = next(p for p in loaded.model_packages if p.package_id == "pkgtrt")
    assert trt_package.trt_package_details.same_cc_compatible is True
    assert trt_package.environment_requirements.l4t_version == Version("36.3")


def test_alias_resolution(registry_home) -> None:
    # given
    metadata = _example_metadata(model_id="workspace/model/1")

    # when
    offline_registry.record_successful_load(
        model_metadata=metadata,
        requested_model_id="my-alias",
        proven_package_id="pkgonnx",
    )
    by_alias = offline_registry.load_model_metadata(model_id="my-alias")
    record = offline_registry.load_record_raw(model_id="my-alias")

    # then
    assert by_alias is not None
    assert by_alias.model_id == "workspace/model/1"
    assert record["requested_aliases"] == ["my-alias"]


def test_merge_preserves_existing_packages_and_proven(registry_home) -> None:
    # given
    metadata = _example_metadata()
    only_trt = ModelMetadata(
        model_id=metadata.model_id,
        model_architecture=metadata.model_architecture,
        task_type=metadata.task_type,
        model_packages=[metadata.model_packages[1]],
    )

    # when
    offline_registry.record_successful_load(
        model_metadata=metadata,
        requested_model_id=metadata.model_id,
        proven_package_id="pkgonnx",
    )
    offline_registry.record_successful_load(
        model_metadata=only_trt,
        requested_model_id=metadata.model_id,
        proven_package_id="pkgtrt",
    )
    record = offline_registry.load_record_raw(model_id=metadata.model_id)

    # then
    assert {p["package_id"] for p in record["packages"]} == {"pkgonnx", "pkgtrt"}
    assert set(record["proven"].keys()) == {"pkgonnx", "pkgtrt"}


def test_unknown_fields_are_tolerated_and_newer_format_skipped(registry_home) -> None:
    # given
    metadata = _example_metadata()
    offline_registry.record_successful_load(
        model_metadata=metadata,
        requested_model_id=metadata.model_id,
        proven_package_id="pkgonnx",
    )
    record_path = offline_registry._record_path(
        canonical_model_id=metadata.model_id
    )
    content = json.load(open(record_path))
    content["some_future_field"] = {"nested": True}
    json.dump(content, open(record_path, "w"))

    # when
    tolerated = offline_registry.load_model_metadata(model_id=metadata.model_id)
    content["format_version"] = offline_registry.REGISTRY_FORMAT_VERSION + 1
    json.dump(content, open(record_path, "w"))
    skipped = offline_registry.load_model_metadata(model_id=metadata.model_id)

    # then
    assert tolerated is not None
    assert skipped is None


def test_record_refused_when_proven_package_has_no_identity(registry_home) -> None:
    # given: the proven package has an unhashed artefact that is not materialized
    metadata = ModelMetadata(
        model_id="workspace/model/2",
        model_architecture="rfdetr",
        task_type="object-detection",
        model_packages=[
            ModelPackageMetadata(
                package_id="pkgunhashed",
                backend=BackendType.ONNX,
                package_artefacts=[
                    FileDownloadSpecs(
                        download_url="https://signed.example/file.bin",
                        file_handle="file.bin",
                        md5_hash=None,
                    ),
                ],
            )
        ],
    )

    # when
    registered = offline_registry.record_successful_load(
        model_metadata=metadata,
        requested_model_id="workspace/model/2",
        proven_package_id="pkgunhashed",
    )

    # then
    assert registered is False
    assert offline_registry.load_record_raw(model_id="workspace/model/2") is None


def test_purge_record(registry_home) -> None:
    # given
    metadata = _example_metadata()
    offline_registry.record_successful_load(
        model_metadata=metadata,
        requested_model_id="my-alias",
        proven_package_id="pkgonnx",
    )

    # when
    purged = offline_registry.purge_record(model_id="my-alias")

    # then
    assert purged is True
    assert offline_registry.load_record_raw(model_id=metadata.model_id) is None
    assert offline_registry.purge_record(model_id=metadata.model_id) is False


def test_verify_records_reports_missing_packages(registry_home) -> None:
    # given: recorded artefacts were never materialized on disk
    metadata = _example_metadata()
    offline_registry.record_successful_load(
        model_metadata=metadata,
        requested_model_id=metadata.model_id,
        proven_package_id="pkgonnx",
    )

    # when
    results = offline_registry.verify_records(model_id=metadata.model_id)

    # then
    assert len(results) == 3
    assert {result["status"] for result in results} == {"missing"}


def _record_in_subprocess(inference_home: str, package_index: int) -> None:
    from unittest import mock as _mock

    from inference_models.weights_providers import offline_registry as _registry

    with _mock.patch.object(_registry, "INFERENCE_HOME", inference_home):
        metadata = _example_metadata()
        only_one = ModelMetadata(
            model_id=metadata.model_id,
            model_architecture=metadata.model_architecture,
            task_type=metadata.task_type,
            model_packages=[metadata.model_packages[package_index]],
        )
        for _ in range(20):
            _registry.record_successful_load(
                model_metadata=only_one,
                requested_model_id=metadata.model_id,
                proven_package_id=only_one.model_packages[0].package_id,
            )


def test_concurrent_appends_lose_neither_write(registry_home) -> None:
    # given
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(
            target=_record_in_subprocess, args=(str(registry_home), index)
        )
        for index in (0, 1)
    ]

    # when
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)
        assert worker.exitcode == 0

    # then
    record = offline_registry.load_record_raw(model_id="workspace/model/1")
    assert {p["package_id"] for p in record["packages"]} == {"pkgonnx", "pkgtrt"}
    assert set(record["proven"].keys()) == {"pkgonnx", "pkgtrt"}
