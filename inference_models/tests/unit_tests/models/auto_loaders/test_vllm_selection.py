import importlib
import warnings
from pathlib import Path
from typing import List
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference_models import _offline, configuration
from inference_models.errors import InvalidEnvVariable
from inference_models.models.auto_loaders import core
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import (
    REGISTERED_MODELS,
    VLM_TASK,
)
from inference_models.models.vllm_proxy import adapter_manager as adapter_manager_module
from inference_models.models.vllm_proxy import qwen3_5_vllm as qwen3_5_vllm_module
from inference_models.models.vllm_proxy import qwen3_8_vllm as qwen3_8_vllm_module
from inference_models.models.vllm_proxy import qwen3vl_vllm as qwen3vl_vllm_module
from inference_models.models.vllm_proxy.adapter_manager import AdapterManager
from inference_models.models.vllm_proxy.qwen3_5_vllm import Qwen35VLLMProxy
from inference_models.models.vllm_proxy.qwen3_8_vllm import Qwen38VLLMProxy
from inference_models.models.vllm_proxy.qwen3vl_vllm import Qwen3VLVLLMProxy
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    ModelMetadata,
    ModelPackageMetadata,
)
from tests.unit_tests.models.vllm_proxy.common import write_adapter_package


def build_metadata(
    model_id: str = "ws/proj/1",
    model_architecture: str = "qwen3_5",
    task_type: str = VLM_TASK,
    model_variant: str = "qwen3_5-0.8b",
    with_adapter_files: bool = True,
) -> ModelMetadata:
    artefacts: List[FileDownloadSpecs] = [
        FileDownloadSpecs(
            download_url="https://weights.example/base/model.safetensors",
            file_handle="base/model.safetensors",
            md5_hash="ff01",
        ),
    ]
    if with_adapter_files:
        artefacts.extend(
            [
                FileDownloadSpecs(
                    download_url="https://weights.example/adapter_config.json",
                    file_handle="adapter_config.json",
                    md5_hash="aa01",
                ),
                FileDownloadSpecs(
                    download_url="https://weights.example/adapter_model.safetensors",
                    file_handle="adapter_model.safetensors",
                    md5_hash="aa02",
                ),
            ]
        )
    return ModelMetadata(
        model_id=model_id,
        model_architecture=model_architecture,
        model_packages=[
            ModelPackageMetadata(
                package_id="pkg1",
                backend=BackendType.HF,
                package_artefacts=artefacts,
            )
        ],
        task_type=task_type,
        model_variant=model_variant,
    )


@pytest.fixture
def vllm_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(configuration, "VLLM_PROXY_ENABLED", True)
    monkeypatch.setattr(configuration, "INFERENCE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(configuration, "VLLM_SERVED_BASE_VARIANT", "qwen3_5-0.8b")
    monkeypatch.setattr(configuration, "VLLM_SERVED_BASE_NAME", "qwen3_5-0.8b")
    monkeypatch.setattr(configuration, "VLLM_DORA_POLICY", "reject")


@pytest.fixture
def adapter_downloads(monkeypatch) -> list:
    calls = []

    def _fake_download(target_dir: str, files_specs, verbose=True, **kwargs):
        calls.append(sorted(handle for handle, _, _ in files_specs))
        write_adapter_package(target_dir=target_dir)

    monkeypatch.setattr(
        adapter_manager_module, "download_files_to_directory", _fake_download
    )
    return calls


@pytest.fixture
def adapter_manager(monkeypatch) -> AdapterManager:
    manager = AdapterManager(client=MagicMock())
    for module in (qwen3_5_vllm_module, qwen3vl_vllm_module, qwen3_8_vllm_module):
        monkeypatch.setattr(module, "get_adapter_manager", lambda: manager)
    return manager


def test_backend_type_has_vllm_member() -> None:
    assert BackendType("vllm") is BackendType.VLLM


@pytest.mark.parametrize(
    "architecture, expected_class",
    [
        ("qwen3vl", Qwen3VLVLLMProxy),
        ("qwen3_5", Qwen35VLLMProxy),
        ("qwen3_8", Qwen38VLLMProxy),
    ],
)
def test_registry_maps_qwen_architectures_to_proxy_classes(
    architecture: str, expected_class: type
) -> None:
    assert (
        REGISTERED_MODELS[(architecture, VLM_TASK, BackendType.VLLM)].resolve()
        is expected_class
    )


def test_flag_on_returns_proxy_without_negotiation_or_package_download(
    vllm_pool, adapter_downloads, adapter_manager: AdapterManager
) -> None:
    metadata = build_metadata()

    with mock.patch.object(
        core, "get_model_from_provider", return_value=metadata
    ) as provider, mock.patch.object(
        core, "negotiate_model_packages"
    ) as negotiate, mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache"
    ) as cache_load, mock.patch.object(
        core, "download_files_to_directory"
    ) as package_download:
        result = core.AutoModel.from_pretrained(
            "ws/proj/1",
            api_key="tenant-key",
            weights_provider_extra_headers={"X-Extra": "1"},
        )

    assert isinstance(result, Qwen35VLLMProxy)
    assert result.model_id == "ws/proj/1"
    assert result.api_key == "tenant-key"
    provider.assert_called_once_with(
        provider="roboflow",
        model_id="ws/proj/1",
        api_key="tenant-key",
        weights_provider_extra_query_params=None,
        weights_provider_extra_headers={"X-Extra": "1"},
    )
    negotiate.assert_not_called()
    cache_load.assert_not_called()
    package_download.assert_not_called()
    assert adapter_downloads == [["adapter_config.json", "adapter_model.safetensors"]]
    adapter_manager.client.load_lora_adapter.assert_called_once()
    assert adapter_manager.get_registration(result._served_name) is not None


def test_flag_on_forwards_metadata_credentials_and_init_kwargs_to_proxy(
    vllm_pool,
) -> None:
    metadata = build_metadata(model_architecture="qwen3vl")
    sentinel = object()

    with mock.patch.object(
        core, "get_model_from_provider", return_value=metadata
    ), mock.patch.object(
        Qwen3VLVLLMProxy, "from_model_metadata", return_value=sentinel
    ) as from_model_metadata:
        result = core.AutoModel.from_pretrained(
            "ws/proj/1",
            api_key="tenant-key",
            weights_provider_extra_headers={"X-Extra": "1"},
            device="cpu",
            custom_option="kept",
        )

    assert result is sentinel
    from_model_metadata.assert_called_once()
    call_kwargs = from_model_metadata.call_args.kwargs
    assert call_kwargs["model_id"] == "ws/proj/1"
    assert call_kwargs["metadata"] is metadata
    assert call_kwargs["api_key"] == "tenant-key"
    assert call_kwargs["weights_provider_extra_headers"] == {"X-Extra": "1"}
    assert call_kwargs["custom_option"] == "kept"
    assert str(call_kwargs["device"]) == "cpu"


def test_flag_on_base_model_id_short_circuits_without_adapter_registration(
    vllm_pool, adapter_downloads, adapter_manager: AdapterManager
) -> None:
    metadata = build_metadata(model_id="qwen3_5-0.8b", with_adapter_files=False)

    with mock.patch.object(
        core, "get_model_from_provider", return_value=metadata
    ), mock.patch.object(core, "negotiate_model_packages") as negotiate:
        result = core.AutoModel.from_pretrained("qwen3_5-0.8b", api_key="key")

    assert isinstance(result, Qwen35VLLMProxy)
    assert result._served_name == "qwen3_5-0.8b"
    negotiate.assert_not_called()
    adapter_manager.client.load_lora_adapter.assert_not_called()
    assert adapter_downloads == []


def test_flag_off_takes_hf_path_through_negotiation(monkeypatch) -> None:
    monkeypatch.setattr(configuration, "VLLM_PROXY_ENABLED", False)
    metadata = build_metadata()
    expected_model = MagicMock()

    with mock.patch.object(
        core, "get_model_from_provider", return_value=metadata
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ), mock.patch.object(
        core, "negotiate_model_packages", return_value=[]
    ) as negotiate, mock.patch.object(
        core, "attempt_loading_matching_model_packages", return_value=expected_model
    ):
        result = core.AutoModel.from_pretrained(
            "ws/proj/1", api_key="key", use_auto_resolution_cache=False
        )

    assert result is expected_model
    negotiate.assert_called_once()
    assert negotiate.call_args.kwargs["model_architecture"] == "qwen3_5"


def test_flag_on_non_qwen_architecture_falls_through_with_single_metadata_fetch(
    vllm_pool,
) -> None:
    metadata = build_metadata(
        model_architecture="yolov8",
        task_type="object-detection",
        model_variant=None,
        with_adapter_files=False,
    )
    expected_model = MagicMock()

    with mock.patch.object(
        core, "get_model_from_provider", return_value=metadata
    ) as provider, mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ), mock.patch.object(
        core, "negotiate_model_packages", return_value=[]
    ) as negotiate, mock.patch.object(
        core, "attempt_loading_matching_model_packages", return_value=expected_model
    ):
        result = core.AutoModel.from_pretrained(
            "ws/proj/1", api_key="key", use_auto_resolution_cache=False
        )

    assert result is expected_model
    negotiate.assert_called_once()
    assert provider.call_count == 1


def test_flag_on_non_qwen_architecture_with_warm_up_fetches_metadata_once(
    vllm_pool, monkeypatch
) -> None:
    monkeypatch.setattr(core, "OFFLINE_MODE_WARM_UP", True)
    metadata = build_metadata(
        model_architecture="yolov8",
        task_type="object-detection",
        model_variant=None,
        with_adapter_files=False,
    )
    expected_model = MagicMock()

    with mock.patch.object(
        core, "get_model_from_provider", return_value=metadata
    ) as provider, mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ), mock.patch.object(
        core, "negotiate_model_packages", return_value=[]
    ), mock.patch.object(
        core, "attempt_loading_matching_model_packages", return_value=expected_model
    ):
        result = core.AutoModel.from_pretrained(
            "ws/proj/1", api_key="key", use_auto_resolution_cache=False
        )

    assert result is expected_model
    assert provider.call_count == 1


def test_flag_on_local_path_never_touches_the_provider(
    vllm_pool, tmp_path: Path
) -> None:
    local_model_dir = tmp_path / "local-model"
    local_model_dir.mkdir()
    expected_model = MagicMock()

    with mock.patch.object(
        core, "get_model_from_provider"
    ) as provider, mock.patch.object(
        core, "attempt_loading_model_from_local_storage", return_value=expected_model
    ):
        result = core.AutoModel.from_pretrained(str(local_model_dir))

    assert result is expected_model
    provider.assert_not_called()


def test_offline_mode_with_flag_keeps_the_package_importable(monkeypatch) -> None:
    monkeypatch.setattr(_offline, "OFFLINE_MODE", True)
    monkeypatch.setenv("VLLM_PROXY_ENABLED", "true")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reloaded = importlib.reload(configuration)
        assert reloaded.VLLM_PROXY_ENABLED is True
        with pytest.raises(
            InvalidEnvVariable,
            match="VLLM_PROXY_ENABLED is not supported while OFFLINE_MODE is enabled",
        ):
            reloaded.validate_vllm_proxy_settings()
    finally:
        monkeypatch.undo()
        importlib.reload(configuration)
    assert configuration.OFFLINE_MODE is _offline.OFFLINE_MODE


def test_offline_mode_with_flag_rejects_remote_loads_when_the_proxy_is_selected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(configuration, "OFFLINE_MODE", True)
    monkeypatch.setattr(configuration, "VLLM_PROXY_ENABLED", True)

    with mock.patch.object(core, "get_model_from_provider") as provider:
        with pytest.raises(InvalidEnvVariable, match="OFFLINE_MODE"):
            core.AutoModel.from_pretrained("qwen3_5-0.8b", api_key="key")

    provider.assert_not_called()
