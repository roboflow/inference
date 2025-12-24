import os.path
import re
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from requests_mock import Mocker

from inference_models import AutoModel, ClassificationPrediction
from inference_models.configuration import ROBOFLOW_API_HOST
from inference_models.errors import (
    DirectLocalStorageAccessError,
    ModelLoadingError,
    UnauthorizedModelAccessError,
)
from inference_models.models.auto_loaders import auto_resolution_cache, core
from inference_models.models.auto_loaders.access_manager import (
    AccessIdentifiers,
    ModelAccessManager,
)
from inference_models.models.auto_loaders.entities import AnyModel


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_auto_loading_when_model_access_is_forbidden() -> None:
    # given
    model_access_manager = MagicMock()
    model_access_manager.is_model_access_forbidden.return_value = True

    # when
    with pytest.raises(UnauthorizedModelAccessError):
        AutoModel.from_pretrained(
            "yolov8n-640", model_access_manager=model_access_manager
        )


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_auto_loading_from_local_model_package_with_custom_code(
    example_model_package_dir: str,
) -> None:
    # when
    model = AutoModel.from_pretrained(example_model_package_dir)

    # then
    assert isinstance(model(np.zeros((192, 168, 3))), ClassificationPrediction)


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_auto_loading_from_local_model_package_with_custom_code_when_local_code_packages_not_allowed(
    example_model_package_dir: str,
) -> None:
    # when
    with pytest.raises(ModelLoadingError):
        _ = AutoModel.from_pretrained(
            example_model_package_dir, allow_local_code_packages=False
        )


@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_auto_loading_from_local_model_package_with_custom_code_when_direct_local_storage_loading_not_allowed(
    example_model_package_dir: str,
) -> None:
    # when
    with pytest.raises(DirectLocalStorageAccessError):
        _ = AutoModel.from_pretrained(
            example_model_package_dir, allow_direct_local_storage_loading=False
        )


MOCKED_WEIGHTS_PROVIDER_RESPONSE = {
    "status": "ok",
    "modelMetadata": {
        "type": "external-model-metadata-v1",
        "modelId": "microsoft-coco-obj-det/8",
        "modelArchitecture": "yolov8",
        "taskType": "object-detection",
        "nextPage": None,
        "modelPackages": [
            {
                "type": "external-model-package-v1",
                "packageId": "8a530473c3fd5a17df7b46abf7271442",
                "packageManifest": {
                    "type": "trt-model-package-v1",
                    "backendType": "trt",
                    "dynamicBatchSize": True,
                    "quantization": "fp16",
                    "cudaDeviceType": "orin",
                    "cudaDeviceCC": "8.7",
                    "cudaVersion": "12.6",
                    "trtVersion": "10.3.0.30.post1",
                    "machineType": "jetson",
                    "sameCCCompatible": True,
                    "minBatchSize": 1,
                    "optBatchSize": 8,
                    "maxBatchSize": 16,
                    "machineSpecs": {
                        "type": "jetson-machine-specs-v1",
                        "driverVersion": "540.4.0",
                        "l4tVersion": "32.4.0",  # picked non-existing version such that the package won't be selected
                        "deviceName": "nvidia-jetson-orin-nx",
                    },
                },
                "packageFiles": [
                    {
                        "fileHandle": "trt_config.json",
                        "downloadUrl": "https://storage.googleapis.com/roboflow-tests-assets/yolov8n-packages/package-jetson/trt_config.json",
                        "md5Hash": "164913b5bf5a5743f204f2646386b69e",
                    },
                    {
                        "fileHandle": "engine.plan",
                        "downloadUrl": "https://storage.googleapis.com/roboflow-tests-assets/yolov8n-packages/package-jetson/engine.plan",
                        "md5Hash": "717aacb6db067a106709ff854e924662",
                    },
                    {
                        "fileHandle": "environment.json",
                        "downloadUrl": "https://storage.googleapis.com/roboflow-tests-assets/yolov8n-packages/package-jetson/environment.json",
                        "md5Hash": "72d83c60ba2f3783b9fa0aaeea5a6e27",
                    },
                    {
                        "fileHandle": "class_names.txt",
                        "downloadUrl": "https://storage.googleapis.com/roboflow-tests-assets/yolov8n-packages/package-jetson/class_names.txt",
                        "md5Hash": "571d630ea11ec39c0b6e6e9ff216b151",
                    },
                ],
                "trustedSource": True,
            },
            {
                "type": "external-model-package-v1",
                "packageId": "8fdaa3c2756a28aa404901ebcb6492bd",
                "packageManifest": {
                    "type": "onnx-model-package-v1",
                    "backendType": "onnx",
                    "dynamicBatchSize": False,
                    "staticBatchSize": 1,
                    "quantization": "fp32",
                    "opset": 16,
                    "incompatibleProviders": None,
                },
                "packageFiles": [
                    {
                        "fileHandle": "environment.json",
                        "downloadUrl": "https://storage.googleapis.com/roboflow-tests-assets/yolov8n-packages/package-base-onnx/environment.json",
                        "md5Hash": "72d83c60ba2f3783b9fa0aaeea5a6e27",
                    },
                    {
                        "fileHandle": "weights.onnx",
                        "downloadUrl": "https://storage.googleapis.com/roboflow-tests-assets/yolov8n-packages/package-base-onnx/weights.onnx",
                        "md5Hash": "205c4eceff850318d3b8383cca433540",
                    },
                    {
                        "fileHandle": "class_names.txt",
                        "downloadUrl": "https://storage.googleapis.com/roboflow-tests-assets/yolov8n-packages/package-base-onnx/class_names.txt",
                        "md5Hash": "571d630ea11ec39c0b6e6e9ff216b151",
                    },
                ],
                "trustedSource": True,
            },
        ],
    },
}


class AccumulativeModelAccessManager(ModelAccessManager):

    def __init__(self):
        self.on_model_access_forbidden_calls = []
        self.on_model_package_access_granted_calls = []
        self.on_file_created_calls = []
        self.on_file_renamed_calls = []
        self.on_symlink_created_calls = []
        self.on_symlink_deleted_calls = []
        self.on_file_deleted_calls = []
        self.on_directory_deleted_calls = []

    def on_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> None:
        self.on_model_access_forbidden_calls.append(
            {"model_id": model_id, "api_key": api_key}
        )

    def on_model_package_access_granted(
        self, access_identifiers: AccessIdentifiers
    ) -> None:
        self.on_model_package_access_granted_calls.append(
            {"access_identifiers": access_identifiers}
        )

    def on_file_created(
        self, file_path: str, access_identifiers: AccessIdentifiers
    ) -> None:
        self.on_file_created_calls.append(
            {"file_path": file_path, "access_identifiers": access_identifiers}
        )

    def on_file_renamed(
        self, old_path: str, new_path: str, access_identifiers: AccessIdentifiers
    ) -> None:
        self.on_file_renamed_calls.append(
            {
                "old_path": old_path,
                "new_path": new_path,
                "access_identifiers": access_identifiers,
            }
        )

    def on_symlink_created(
        self, target_path: str, link_name: str, access_identifiers: AccessIdentifiers
    ) -> None:
        self.on_symlink_created_calls.append(
            {
                "target_path": target_path,
                "link_name": link_name,
                "access_identifiers": access_identifiers,
            }
        )

    def on_symlink_deleted(self, link_name: str) -> None:
        self.on_symlink_deleted_calls.append(
            {
                "link_name": link_name,
            }
        )

    def on_file_deleted(self, file_path: str) -> None:
        self.on_file_deleted_calls.append({"file_path": file_path})

    def on_directory_deleted(self, dir_path: str) -> None:
        self.on_directory_deleted_calls.append({"dir_path": dir_path})

    def is_model_access_forbidden(self, model_id: str, api_key: Optional[str]) -> bool:
        params = {"model_id": model_id, "api_key": api_key}
        return any(p == params for p in self.on_model_access_forbidden_calls)

    def is_model_package_access_granted(
        self, model_id: str, package_id: str, api_key: Optional[str]
    ) -> bool:
        access_identifiers = AccessIdentifiers(
            model_id=model_id, package_id=package_id, api_key=api_key
        )
        return any(
            p["access_identifiers"] == access_identifiers
            for p in self.on_model_package_access_granted_calls
        )

    def retrieve_model_instance(
        self,
        model_id: str,
        package_id: Optional[str],
        api_key: Optional[str],
        loading_parameter_digest: Optional[str],
    ) -> Optional[AnyModel]:
        return None

    def on_model_loaded(
        self,
        model: AnyModel,
        access_identifiers: AccessIdentifiers,
        model_storage_path: str,
    ) -> None:
        pass

    def on_model_alias_discovered(self, alias: str, model_id: str) -> None:
        pass

    def on_model_dependency_discovered(
        self, base_model_id: str, dependent_model_id: str
    ) -> None:
        pass


@pytest.mark.timeout(60)
@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
@pytest.mark.skip("Skipping for now - registered packages must be fixed")
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
@mock.patch.object(core, "generate_model_package_cache_path")
@mock.patch.object(core, "generate_shared_blobs_path")
def test_auto_loading_with_weights_provider_in_base_scenario(
    generate_shared_blobs_path_mock: MagicMock,
    generate_model_package_cache_path_mock: MagicMock,
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
    requests_mock: Mocker,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_onnx import (
        YOLOv8ForObjectDetectionOnnx,
    )

    storage_manager = AccumulativeModelAccessManager()
    shared_blobs_dir_path = os.path.join(empty_local_dir, "shared-blobs")
    generate_shared_blobs_path_mock.side_effect = lambda: shared_blobs_dir_path
    generate_model_package_cache_path_mock.side_effect = lambda model_id, package_id: (
        os.path.join(empty_local_dir, "models-cache", model_id, package_id)
    )
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: (
            os.path.join(
                empty_local_dir,
                "auto-negotiation-cache",
                f"{auto_negotiation_hash}.json",
            )
        )
    )
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {
                "status_code": 200,
                "json": MOCKED_WEIGHTS_PROVIDER_RESPONSE,
            },
        ],
    )
    requests_mock.get(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )

    # when
    model = AutoModel.from_pretrained(
        "yolov8n-640", model_access_manager=storage_manager
    )

    # then
    expected_access_identifiers = AccessIdentifiers(
        model_id="yolov8n-640",
        package_id="8fdaa3c2756a28aa404901ebcb6492bd",
        api_key=None,
    )
    assert isinstance(model, YOLOv8ForObjectDetectionOnnx)
    assert len(storage_manager.on_model_access_forbidden_calls) == 0
    assert (
        len(storage_manager.on_file_created_calls) == 5
    ), "Expected to create 3 model files from download, one model config and one cache entry"
    assert len(storage_manager.on_model_package_access_granted_calls) == 2
    expected_model_package_dir = os.path.join(
        empty_local_dir,
        "models-cache",
        "yolov8n-640",
        "8fdaa3c2756a28aa404901ebcb6492bd",
    )
    all_symlinks = [p["link_name"] for p in storage_manager.on_symlink_created_calls]
    assert len(all_symlinks) == 3
    shared_blobs_dir_content = [
        e for e in os.listdir(shared_blobs_dir_path) if not e.endswith(".lock")
    ]
    assert len(shared_blobs_dir_content) == 3
    assert set(shared_blobs_dir_content) == {
        "72d83c60ba2f3783b9fa0aaeea5a6e27",
        "205c4eceff850318d3b8383cca433540",
        "571d630ea11ec39c0b6e6e9ff216b151",
    }
    assert {
        e for e in os.listdir(expected_model_package_dir) if not e.endswith(".lock")
    } == {"class_names.txt", "weights.onnx", "environment.json", "model_config.json"}


@pytest.mark.timeout(60)
@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
@pytest.mark.skip("Skipping for now - registered packages must be fixed")
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
@mock.patch.object(core, "generate_model_package_cache_path")
@mock.patch.object(core, "generate_shared_blobs_path")
def test_auto_loading_with_weights_provider_when_cache_for_the_exact_model_and_api_key_filled(
    generate_shared_blobs_path_mock: MagicMock,
    generate_model_package_cache_path_mock: MagicMock,
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
    requests_mock: Mocker,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_onnx import (
        YOLOv8ForObjectDetectionOnnx,
    )

    storage_manager = AccumulativeModelAccessManager()
    shared_blobs_dir_path = os.path.join(empty_local_dir, "shared-blobs")
    generate_shared_blobs_path_mock.side_effect = lambda: shared_blobs_dir_path
    generate_model_package_cache_path_mock.side_effect = lambda model_id, package_id: (
        os.path.join(empty_local_dir, "models-cache", model_id, package_id)
    )
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: (
            os.path.join(
                empty_local_dir,
                "auto-negotiation-cache",
                f"{auto_negotiation_hash}.json",
            )
        )
    )
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {
                "status_code": 200,
                "json": MOCKED_WEIGHTS_PROVIDER_RESPONSE,
            },
        ],
    )
    requests_mock.get(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )
    requests_mock.head(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )

    # when
    _ = AutoModel.from_pretrained("yolov8n-640", model_access_manager=storage_manager)
    storage_manager.on_file_created_calls = []
    model = AutoModel.from_pretrained(
        "yolov8n-640", model_access_manager=storage_manager
    )

    # then
    assert isinstance(model, YOLOv8ForObjectDetectionOnnx)
    assert len(storage_manager.on_model_access_forbidden_calls) == 0
    assert (
        len(storage_manager.on_file_created_calls) == 0
    ), "Expected everything already in cache"


@pytest.mark.timeout(60)
@pytest.mark.slow
@pytest.mark.cpu_only
@pytest.mark.onnx_extras
@pytest.mark.skip("Skipping for now - registered packages must be fixed")
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
@mock.patch.object(core, "generate_model_package_cache_path")
@mock.patch.object(core, "generate_shared_blobs_path")
def test_auto_loading_with_weights_provider_when_cache_for_the_exact_model_but_different_api_key_filled(
    generate_shared_blobs_path_mock: MagicMock,
    generate_model_package_cache_path_mock: MagicMock,
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
    requests_mock: Mocker,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_onnx import (
        YOLOv8ForObjectDetectionOnnx,
    )

    storage_manager = AccumulativeModelAccessManager()
    shared_blobs_dir_path = os.path.join(empty_local_dir, "shared-blobs")
    generate_shared_blobs_path_mock.side_effect = lambda: shared_blobs_dir_path
    generate_model_package_cache_path_mock.side_effect = lambda model_id, package_id: (
        os.path.join(empty_local_dir, "models-cache", model_id, package_id)
    )
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: (
            os.path.join(
                empty_local_dir,
                "auto-negotiation-cache",
                f"{auto_negotiation_hash}.json",
            )
        )
    )
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {
                "status_code": 200,
                "json": MOCKED_WEIGHTS_PROVIDER_RESPONSE,
            },
        ],
    )
    requests_mock.get(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )
    requests_mock.head(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )

    # when
    _ = AutoModel.from_pretrained("yolov8n-640", model_access_manager=storage_manager)
    storage_manager.on_file_created_calls = []
    model = AutoModel.from_pretrained(
        "yolov8n-640", api_key="different", model_access_manager=storage_manager
    )

    # then
    assert isinstance(model, YOLOv8ForObjectDetectionOnnx)
    assert len(storage_manager.on_model_access_forbidden_calls) == 0
    assert (
        len(storage_manager.on_file_created_calls) == 1
    ), "Expected to create new entry in cache"
    assert (
        len(
            [
                e
                for e in os.listdir(
                    os.path.join(empty_local_dir, "auto-negotiation-cache")
                )
                if not e.endswith(".lock")
            ]
        )
        == 2
    )


@pytest.mark.timeout(60)
@pytest.mark.cpu_only
@pytest.mark.slow
@pytest.mark.skip("Skipping for now - registered packages must be fixed")
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
@mock.patch.object(core, "generate_model_package_cache_path")
@mock.patch.object(core, "generate_shared_blobs_path")
def test_auto_loading_with_weights_provider_when_api_denoted_forbidden(
    generate_shared_blobs_path_mock: MagicMock,
    generate_model_package_cache_path_mock: MagicMock,
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
    requests_mock: Mocker,
) -> None:
    # given
    storage_manager = AccumulativeModelAccessManager()
    shared_blobs_dir_path = os.path.join(empty_local_dir, "shared-blobs")
    generate_shared_blobs_path_mock.side_effect = lambda: shared_blobs_dir_path
    generate_model_package_cache_path_mock.side_effect = lambda model_id, package_id: (
        os.path.join(empty_local_dir, "models-cache", model_id, package_id)
    )
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: (
            os.path.join(
                empty_local_dir,
                "auto-negotiation-cache",
                f"{auto_negotiation_hash}.json",
            )
        )
    )
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {
                "status_code": 401,
            },
        ],
    )
    requests_mock.get(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )
    requests_mock.head(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )

    # when
    with pytest.raises(UnauthorizedModelAccessError):
        _ = AutoModel.from_pretrained(
            "yolov8n-640", model_access_manager=storage_manager
        )
    with pytest.raises(UnauthorizedModelAccessError):
        _ = AutoModel.from_pretrained(
            "yolov8n-640", model_access_manager=storage_manager
        )

    # then
    assert (
        len(storage_manager.on_model_access_forbidden_calls) == 1
    ), "Only first attempt should be denoted"
    assert len(storage_manager.on_file_created_calls) == 0


@pytest.mark.timeout(60)
@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
@pytest.mark.skip("Skipping for now - registered packages must be fixed")
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
@mock.patch.object(core, "generate_model_package_cache_path")
@mock.patch.object(core, "generate_shared_blobs_path")
def test_auto_loading_with_weights_provider_when_api_denoted_forbidden_for_one_key_but_allowed_for_another(
    generate_shared_blobs_path_mock: MagicMock,
    generate_model_package_cache_path_mock: MagicMock,
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
    requests_mock: Mocker,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_onnx import (
        YOLOv8ForObjectDetectionOnnx,
    )

    storage_manager = AccumulativeModelAccessManager()
    shared_blobs_dir_path = os.path.join(empty_local_dir, "shared-blobs")
    generate_shared_blobs_path_mock.side_effect = lambda: shared_blobs_dir_path
    generate_model_package_cache_path_mock.side_effect = lambda model_id, package_id: (
        os.path.join(empty_local_dir, "models-cache", model_id, package_id)
    )
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: (
            os.path.join(
                empty_local_dir,
                "auto-negotiation-cache",
                f"{auto_negotiation_hash}.json",
            )
        )
    )
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {
                "status_code": 200,
                "json": MOCKED_WEIGHTS_PROVIDER_RESPONSE,
            },
            {
                "status_code": 401,
            },
        ],
    )
    requests_mock.get(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )
    requests_mock.head(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )

    # when
    model = AutoModel.from_pretrained(
        "yolov8n-640", api_key="another", model_access_manager=storage_manager
    )
    with pytest.raises(UnauthorizedModelAccessError):
        _ = AutoModel.from_pretrained(
            "yolov8n-640", model_access_manager=storage_manager
        )

    # then
    expected_access_identifiers = AccessIdentifiers(
        model_id="yolov8n-640",
        package_id="8fdaa3c2756a28aa404901ebcb6492bd",
        api_key="another",
    )
    assert isinstance(model, YOLOv8ForObjectDetectionOnnx)
    assert len(storage_manager.on_model_access_forbidden_calls) == 1
    assert (
        len(storage_manager.on_file_created_calls) == 5
    ), "Expected to create 3 model files from download, one model config and one cache entry"
    assert len(storage_manager.on_model_package_access_granted_calls) == 2
    expected_model_package_dir = os.path.join(
        empty_local_dir,
        "models-cache",
        "yolov8n-640",
        "8fdaa3c2756a28aa404901ebcb6492bd",
    )
    all_symlinks = [p["link_name"] for p in storage_manager.on_symlink_created_calls]
    assert len(all_symlinks) == 3
    shared_blobs_dir_content = [
        e for e in os.listdir(shared_blobs_dir_path) if not e.endswith(".lock")
    ]
    assert len(shared_blobs_dir_content) == 3
    assert set(shared_blobs_dir_content) == {
        "72d83c60ba2f3783b9fa0aaeea5a6e27",
        "205c4eceff850318d3b8383cca433540",
        "571d630ea11ec39c0b6e6e9ff216b151",
    }
    assert {
        e for e in os.listdir(expected_model_package_dir) if not e.endswith(".lock")
    } == {"class_names.txt", "weights.onnx", "environment.json", "model_config.json"}


@pytest.mark.timeout(60)
@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
@pytest.mark.skip("Skipping for now - registered packages must be fixed")
@mock.patch.object(auto_resolution_cache, "generate_auto_resolution_cache_path")
@mock.patch.object(core, "generate_model_package_cache_path")
@mock.patch.object(core, "generate_shared_blobs_path")
def test_auto_loading_from_cached_local_path(
    generate_shared_blobs_path_mock: MagicMock,
    generate_model_package_cache_path_mock: MagicMock,
    generate_auto_resolution_cache_path_mock: MagicMock,
    empty_local_dir: str,
    requests_mock: Mocker,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_onnx import (
        YOLOv8ForObjectDetectionOnnx,
    )

    shared_blobs_dir_path = os.path.join(empty_local_dir, "shared-blobs")
    generate_shared_blobs_path_mock.side_effect = lambda: shared_blobs_dir_path
    generate_model_package_cache_path_mock.side_effect = lambda model_id, package_id: (
        os.path.join(empty_local_dir, "models-cache", model_id, package_id)
    )
    generate_auto_resolution_cache_path_mock.side_effect = (
        lambda auto_negotiation_hash: (
            os.path.join(
                empty_local_dir,
                "auto-negotiation-cache",
                f"{auto_negotiation_hash}.json",
            )
        )
    )
    requests_mock.get(
        f"{ROBOFLOW_API_HOST}/models/v1/external/weights",
        [
            {
                "status_code": 200,
                "json": MOCKED_WEIGHTS_PROVIDER_RESPONSE,
            },
        ],
    )
    requests_mock.get(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )
    requests_mock.head(
        re.compile(r"https://storage\.googleapis\.com/.*"), real_http=True
    )

    # when
    _ = AutoModel.from_pretrained("yolov8n-640")
    expected_model_package_dir = os.path.join(
        empty_local_dir,
        "models-cache",
        "yolov8n-640",
        "8fdaa3c2756a28aa404901ebcb6492bd",
    )
    model = AutoModel.from_pretrained(expected_model_package_dir)

    # then

    assert isinstance(model, YOLOv8ForObjectDetectionOnnx)
