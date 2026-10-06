import json
import shutil
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from inference_models import configuration
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.vllm_proxy import adapter_manager as adapter_manager_module
from inference_models.models.vllm_proxy.adapter_manager import (
    AdapterManager,
    normalize_base_variant,
)
from inference_models.models.vllm_proxy.errors import (
    AdapterNotServableError,
    NotServableOnVLLMError,
    VLLMConnectionError,
    VLLMHTTPError,
)
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    ModelMetadata,
    ModelPackageMetadata,
)
from tests.unit_tests.models.vllm_proxy.common import (
    build_adapter_config,
    write_adapter_package,
)


def build_metadata(
    model_id: str = "some-workspace/some-project/1",
    model_variant: Optional[str] = "qwen3_5-0.8b",
    model_architecture: str = "qwen3_5",
    package_id: str = "pkg1",
    backend: BackendType = BackendType.HF,
    with_adapter_files: bool = True,
    adapter_md5_prefix: str = "aa",
) -> ModelMetadata:
    artefacts: List[FileDownloadSpecs] = [
        FileDownloadSpecs(
            download_url="https://weights.example/base/config.json",
            file_handle="base/config.json",
            md5_hash="ff01",
        ),
        FileDownloadSpecs(
            download_url="https://weights.example/base/model.safetensors",
            file_handle="base/model.safetensors",
            md5_hash="ff02",
        ),
    ]
    if with_adapter_files:
        artefacts.extend(
            [
                FileDownloadSpecs(
                    download_url="https://weights.example/adapter_config.json",
                    file_handle="adapter_config.json",
                    md5_hash=f"{adapter_md5_prefix}01",
                ),
                FileDownloadSpecs(
                    download_url="https://weights.example/adapter_model.safetensors",
                    file_handle="adapter_model.safetensors",
                    md5_hash=f"{adapter_md5_prefix}02",
                ),
            ]
        )
    package = ModelPackageMetadata(
        package_id=package_id,
        backend=backend,
        package_artefacts=artefacts,
    )
    return ModelMetadata(
        model_id=model_id,
        model_architecture=model_architecture,
        model_packages=[package],
        task_type="vlm",
        model_variant=model_variant,
    )


def _set_served_base(monkeypatch, variant: str, name: str = None) -> None:
    monkeypatch.setattr(configuration, "VLLM_SERVED_BASE_VARIANT", variant)
    monkeypatch.setattr(configuration, "VLLM_SERVED_BASE_NAME", name or variant)


@pytest.fixture(autouse=True)
def default_vllm_configuration(monkeypatch) -> None:
    _set_served_base(monkeypatch, variant="qwen3_5-0.8b")
    monkeypatch.setattr(configuration, "VLLM_DORA_POLICY", "reject")
    monkeypatch.setattr(configuration, "VLLM_MAX_REGISTERED_ADAPTERS", 64)


@pytest.fixture
def fake_download(monkeypatch):
    """Replaces download_files_to_directory with a fixture-package writer."""
    calls = []

    def _fake_download(target_dir: str, files_specs, verbose=True, **kwargs):
        calls.append((target_dir, list(files_specs)))
        write_adapter_package(target_dir=target_dir)
        return {handle: str(Path(target_dir) / handle) for handle, _, _ in files_specs}

    monkeypatch.setattr(
        adapter_manager_module, "download_files_to_directory", _fake_download
    )
    return calls


@pytest.fixture
def model_cache_dir(monkeypatch, tmp_path):
    cache_dir = str(tmp_path / "cache")
    monkeypatch.setattr(configuration, "INFERENCE_HOME", cache_dir)
    return cache_dir


def _install_provider(monkeypatch, metadata_by_model_id: dict) -> MagicMock:
    provider = MagicMock(
        side_effect=lambda model_id, provider, api_key=None, **kwargs: (
            metadata_by_model_id[model_id]
        )
    )
    monkeypatch.setattr(adapter_manager_module, "get_model_from_provider", provider)
    return provider


def _install_download_with_config(
    monkeypatch, base_model_name_or_path: Optional[str]
) -> list:
    """Installs a download fake writing an adapter declaring the given base."""
    calls = []

    def _fake_download(target_dir: str, files_specs, verbose=True, **kwargs):
        calls.append(target_dir)
        write_adapter_package(
            target_dir=target_dir,
            config=build_adapter_config(
                base_model_name_or_path=base_model_name_or_path
            ),
        )

    monkeypatch.setattr(
        adapter_manager_module, "download_files_to_directory", _fake_download
    )
    return calls


def _install_logger_mock(monkeypatch) -> MagicMock:
    logger_mock = MagicMock()
    monkeypatch.setattr(adapter_manager_module, "LOGGER", logger_mock)
    return logger_mock


def _rendered_warnings(logger_mock: MagicMock) -> List[str]:
    return [
        call.args[0] % tuple(call.args[1:])
        for call in logger_mock.warning.call_args_list
    ]


class TestNormalizeBaseVariant:
    @pytest.mark.parametrize(
        "architecture, variant, expected",
        [
            ("qwen3_5", "qwen3_5-0.8b", "qwen3_5-0.8b"),
            ("qwen3_5", "qwen3_5-0.8b-peft", "qwen3_5-0.8b"),
            ("qwen3_5", "0.8b", "qwen3_5-0.8b"),
            ("qwen3_5", "0.8b-peft", "qwen3_5-0.8b"),
            ("qwen3vl", "2b", "qwen3vl-2b"),
            ("qwen3vl", "2b-peft", "qwen3vl-2b"),
            ("qwen3vl", "qwen3vl-2b-peft", "qwen3vl-2b"),
            ("QWEN3VL", "2B-PEFT", "qwen3vl-2b"),
            ("qwen3vl", "2b-instruct", "qwen3vl-2b-instruct"),
            (None, "2b", None),
            ("qwen3vl", None, None),
            ("", "", None),
        ],
    )
    def test_normalization_matrix(
        self, architecture: Optional[str], variant: Optional[str], expected
    ) -> None:
        assert (
            normalize_base_variant(
                model_architecture=architecture, model_variant=variant
            )
            == expected
        )


class TestVariantMatching:
    """Matrix for the ADVISORY base-variant check of fine-tune metadata.

    Registry `modelVariant` is sometimes misregistered, so a mismatch never
    rejects pre-download - it logs a warning and defers to the adapter's own
    `adapter_config.json` (`cross_check_base_model` in `patch_adapter`).
    """

    @pytest.mark.parametrize(
        "architecture, variant, served_variant",
        [
            ("qwen3_5", "qwen3_5-0.8b", "qwen3_5-0.8b"),
            ("qwen3_5", "0.8b-peft", "qwen3_5-0.8b"),
            ("qwen3_5", "qwen3_5-0.8b-peft", "qwen3_5-0.8b"),
            ("qwen3vl", "2b-peft", "qwen3vl-2b"),
            ("qwen3vl", "2b", "qwen3vl-2b"),
            ("qwen3vl", "2B-PEFT", "qwen3vl-2b"),
        ],
    )
    def test_matching_variant_registers_without_warning(
        self,
        monkeypatch,
        fake_download,
        model_cache_dir,
        architecture: str,
        variant: str,
        served_variant: str,
    ) -> None:
        _set_served_base(monkeypatch, variant=served_variant)
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture=architecture,
            model_variant=variant,
        )
        logger_mock = _install_logger_mock(monkeypatch)
        manager = AdapterManager(client=MagicMock())

        served_name = manager.resolve_and_register(model_id, metadata=metadata)

        assert served_name.startswith("some-workspace-some-project-1-")
        logger_mock.warning.assert_not_called()

    @pytest.mark.parametrize(
        "architecture, variant, served_variant",
        [
            ("qwen3vl", "2b-peft", "qwen3_5-0.8b"),
            ("qwen3_5", "0.8b-peft", "qwen3vl-2b"),
            ("qwen3_5", "qwen3_5-2b", "qwen3_5-0.8b"),
            ("qwen3vl", "4b-peft", "qwen3vl-2b"),
        ],
    )
    def test_mismatching_variant_defers_to_adapter_config(
        self,
        monkeypatch,
        model_cache_dir,
        architecture: str,
        variant: str,
        served_variant: str,
    ) -> None:
        _set_served_base(monkeypatch, variant=served_variant)
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture=architecture,
            model_variant=variant,
        )
        downloads = _install_download_with_config(
            monkeypatch, base_model_name_or_path=f"qwen/{served_variant}"
        )
        logger_mock = _install_logger_mock(monkeypatch)
        client = MagicMock()
        manager = AdapterManager(client=client)

        served_name = manager.resolve_and_register(model_id, metadata=metadata)

        assert served_name.startswith("some-workspace-some-project-1-")
        assert len(downloads) == 1
        client.load_lora_adapter.assert_called_once()
        warnings = _rendered_warnings(logger_mock)
        assert any("deferring to adapter_config" in message for message in warnings)
        assert any("misregistered" in message for message in warnings)


class TestResolveAndRegister:
    def test_base_model_id_returns_served_base_name_without_provider_call(
        self, monkeypatch
    ) -> None:
        provider = _install_provider(monkeypatch, {})
        manager = AdapterManager(client=MagicMock())

        served_name = manager.resolve_and_register("qwen3_5-0.8b")

        assert served_name == "qwen3_5-0.8b"
        provider.assert_not_called()

    def test_passed_metadata_is_used_without_provider_call(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        provider = _install_provider(monkeypatch, {})
        model_id = "some-workspace/some-project/1"
        client = MagicMock()
        manager = AdapterManager(client=client)

        served_name = manager.resolve_and_register(
            model_id, metadata=build_metadata(model_id=model_id), api_key="key"
        )

        assert served_name.startswith("some-workspace-some-project-1-pkg1-")
        provider.assert_not_called()
        client.load_lora_adapter.assert_called_once()

    def test_missing_metadata_is_fetched_from_provider_with_credentials(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        provider = _install_provider(
            monkeypatch, {model_id: build_metadata(model_id=model_id)}
        )
        manager = AdapterManager(client=MagicMock())

        served_name = manager.resolve_and_register(
            model_id,
            api_key="key",
            weights_provider_extra_headers={"X-Extra": "1"},
        )

        assert served_name.startswith("some-workspace-some-project-1-pkg1-")
        provider.assert_called_once_with(
            model_id=model_id,
            provider="roboflow",
            api_key="key",
            weights_provider_extra_headers={"X-Extra": "1"},
        )

    def test_base_package_without_adapter_files_returns_served_base_name(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        metadata = build_metadata(model_id="qwen-base-alias", with_adapter_files=False)
        client = MagicMock()
        manager = AdapterManager(client=client)

        served_name = manager.resolve_and_register("qwen-base-alias", metadata=metadata)

        assert served_name == "qwen3_5-0.8b"
        client.load_lora_adapter.assert_not_called()
        assert fake_download == []

    def test_base_package_with_mismatching_variant_is_not_servable(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        metadata = build_metadata(
            model_id="qwen-base-alias",
            model_variant="qwen3_5-2b",
            with_adapter_files=False,
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        with pytest.raises(NotServableOnVLLMError) as error:
            manager.resolve_and_register("qwen-base-alias", metadata=metadata)

        assert "qwen3_5-2b" in str(error.value)
        assert "qwen3_5-0.8b" in str(error.value)
        client.load_lora_adapter.assert_not_called()
        assert fake_download == []

    def test_fine_tune_is_downloaded_patched_and_registered(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        client = MagicMock()
        manager = AdapterManager(client=client)

        served_name = manager.resolve_and_register(
            model_id, metadata=build_metadata(model_id=model_id), api_key="key"
        )

        assert served_name.startswith("some-workspace-some-project-1-pkg1-")
        client.load_lora_adapter.assert_called_once()
        _, load_kwargs = client.load_lora_adapter.call_args
        assert load_kwargs["name"] == served_name
        assert (Path(load_kwargs["path"]) / "adapter_model.safetensors").is_file()
        assert (Path(load_kwargs["path"]) / "patch_report.json").is_file()
        assert Path(load_kwargs["path"]).is_relative_to(
            Path(model_cache_dir) / "vllm-adapters" / served_name
        )

    def test_only_adapter_files_are_downloaded_never_base_weights(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        manager = AdapterManager(client=MagicMock())

        manager.resolve_and_register(
            model_id, metadata=build_metadata(model_id=model_id)
        )

        assert len(fake_download) == 1
        _, files_specs = fake_download[0]
        downloaded_handles = sorted(handle for handle, _, _ in files_specs)
        assert downloaded_handles == [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]

    def test_slug_includes_content_digest(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        metadata_v1 = build_metadata(model_id=model_id, adapter_md5_prefix="aa")
        metadata_v2 = build_metadata(model_id=model_id, adapter_md5_prefix="bb")
        manager = AdapterManager(client=MagicMock())

        served_v1 = manager.resolve_and_register(model_id, metadata=metadata_v1)
        served_v2 = manager.resolve_and_register(model_id, metadata=metadata_v2)

        assert served_v1 != served_v2

    def test_re_registration_skips_download_but_always_recalls_vllm_load(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id)
        client = MagicMock()
        manager = AdapterManager(client=client)

        served_first = manager.resolve_and_register(model_id, metadata=metadata)
        served_second = manager.resolve_and_register(model_id, metadata=metadata)

        assert served_first == served_second
        assert client.load_lora_adapter.call_count == 2
        for _, load_kwargs in client.load_lora_adapter.call_args_list:
            assert load_kwargs["name"] == served_first
        assert len(fake_download) == 1

    def test_recorded_slug_with_missing_patched_dir_redoes_download_and_patch(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id)
        client = MagicMock()
        manager = AdapterManager(client=client)
        served_name = manager.resolve_and_register(model_id, metadata=metadata)
        shutil.rmtree(manager.get_registration(served_name).patched_dir)

        served_again = manager.resolve_and_register(model_id, metadata=metadata)

        assert served_again == served_name
        assert len(fake_download) == 2
        assert Path(manager.get_registration(served_name).patched_dir).is_dir()

    def test_partial_patched_dir_redoes_download_and_patch(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id)
        manager = AdapterManager(client=MagicMock())
        package = metadata.model_packages[0]
        adapter_files = [
            artefact
            for artefact in package.package_artefacts
            if not artefact.file_handle.startswith("base/")
        ]
        slug = manager._build_slug(
            model_id=metadata.model_id,
            package_id=package.package_id,
            content_digest=manager._compute_content_digest(adapter_files),
        )
        patched_dir = Path(model_cache_dir) / "vllm-adapters" / slug / "patched"
        patched_dir.mkdir(parents=True, exist_ok=True)
        (patched_dir / "adapter_config.json").write_text(json.dumps({"partial": True}))

        served_name = manager.resolve_and_register(model_id, metadata=metadata)

        assert served_name == slug
        assert len(fake_download) == 1
        _, load_kwargs = manager.client.load_lora_adapter.call_args
        assert (Path(load_kwargs["path"]) / "adapter_model.safetensors").is_file()
        assert (Path(load_kwargs["path"]) / "patch_report.json").is_file()

    def test_runtime_svd_policy_is_rejected_before_download(
        self, monkeypatch, model_cache_dir
    ) -> None:
        monkeypatch.setattr(configuration, "VLLM_DORA_POLICY", "svd")
        model_id = "some-workspace/some-project/1"
        download = MagicMock()
        monkeypatch.setattr(
            adapter_manager_module, "download_files_to_directory", download
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        with pytest.raises(NotServableOnVLLMError) as error:
            manager.resolve_and_register(
                model_id, metadata=build_metadata(model_id=model_id)
            )
        assert "VLLM_DORA_POLICY=svd" in str(error.value)
        assert "base/ weights" in str(error.value)
        download.assert_not_called()
        client.load_lora_adapter.assert_not_called()

    def test_dora_adapter_is_rejected_under_default_policy(
        self, monkeypatch, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"

        def _fake_download(target_dir: str, files_specs, verbose=True, **kwargs):
            write_adapter_package(
                target_dir=target_dir, config=build_adapter_config(use_dora=True)
            )

        monkeypatch.setattr(
            adapter_manager_module, "download_files_to_directory", _fake_download
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        with pytest.raises(AdapterNotServableError) as error:
            manager.resolve_and_register(
                model_id, metadata=build_metadata(model_id=model_id)
            )
        assert "DoRA" in str(error.value)
        client.load_lora_adapter.assert_not_called()

    def test_overflow_past_max_registered_warns_and_never_unloads(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        monkeypatch.setattr(configuration, "VLLM_MAX_REGISTERED_ADAPTERS", 2)
        metadata_by_model_id = {
            f"ws/project-{i}/1": build_metadata(
                model_id=f"ws/project-{i}/1",
                package_id=f"pkg{i}",
                adapter_md5_prefix=f"{i}{i}",
            )
            for i in range(3)
        }
        logger_mock = _install_logger_mock(monkeypatch)
        client = MagicMock()
        manager = AdapterManager(client=client)

        served_names = [
            manager.resolve_and_register(model_id, metadata=metadata)
            for model_id, metadata in metadata_by_model_id.items()
        ]

        client.unload_lora_adapter.assert_not_called()
        for served_name in served_names:
            assert manager.get_registration(served_name) is not None
        warnings = _rendered_warnings(logger_mock)
        assert any(
            "exceeds" in message and "VLLM_MAX_REGISTERED_ADAPTERS=2" in message
            for message in warnings
        )

    def test_invalidate_drops_slug_and_next_resolution_reregisters(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id)
        client = MagicMock()
        manager = AdapterManager(client=client)
        served_name = manager.resolve_and_register(model_id, metadata=metadata)

        manager.invalidate(served_name=served_name)

        assert manager.get_registration(served_name) is None
        assert manager.resolve_and_register(model_id, metadata=metadata) == served_name
        assert manager.get_registration(served_name) is not None
        assert client.load_lora_adapter.call_count == 2
        client.unload_lora_adapter.assert_not_called()

    def test_wrong_base_variant_is_rejected_by_adapter_config_cross_check(
        self, monkeypatch, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id, model_variant="qwen3_5-2b")
        downloads = _install_download_with_config(
            monkeypatch, base_model_name_or_path="qwen/qwen3_5-2b"
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        with pytest.raises(AdapterNotServableError) as error:
            manager.resolve_and_register(model_id, metadata=metadata)
        assert len(downloads) == 1
        message = str(error.value)
        assert "qwen/qwen3_5-2b" in message
        assert "qwen3_5-0.8b" in message
        client.load_lora_adapter.assert_not_called()

    @pytest.mark.parametrize("architecture", ["qwen25vl", "florence2", "qwen3_8"])
    def test_wrong_architecture_is_rejected_pre_download(
        self, monkeypatch, architecture: str
    ) -> None:
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id, model_architecture=architecture)
        download = MagicMock()
        monkeypatch.setattr(
            adapter_manager_module, "download_files_to_directory", download
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        with pytest.raises(NotServableOnVLLMError):
            manager.resolve_and_register(model_id, metadata=metadata)
        download.assert_not_called()
        client.load_lora_adapter.assert_not_called()

    def test_missing_hf_package_is_rejected(self, monkeypatch) -> None:
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id, backend=BackendType.ONNX)
        manager = AdapterManager(client=MagicMock())

        with pytest.raises(NotServableOnVLLMError):
            manager.resolve_and_register(model_id, metadata=metadata)

    def test_base_id_matching_served_base_name_short_circuits(
        self, monkeypatch
    ) -> None:
        _set_served_base(monkeypatch, variant="qwen3vl-2b", name="qwen3vl-2b-instruct")
        provider = _install_provider(monkeypatch, {})
        manager = AdapterManager(client=MagicMock())

        served_name = manager.resolve_and_register("qwen3vl-2b-instruct")

        assert served_name == "qwen3vl-2b-instruct"
        provider.assert_not_called()

    def test_base_id_short_circuit_is_case_insensitive(self, monkeypatch) -> None:
        _set_served_base(monkeypatch, variant="qwen3vl-2b", name="qwen3vl-2b-instruct")
        provider = _install_provider(monkeypatch, {})
        manager = AdapterManager(client=MagicMock())

        served_name = manager.resolve_and_register("QWEN3VL-2B")

        assert served_name == "qwen3vl-2b-instruct"
        provider.assert_not_called()

    def test_qwen3vl_fine_tune_is_registered_on_qwen3vl_pool(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        _set_served_base(monkeypatch, variant="qwen3vl-2b", name="qwen3vl-2b-instruct")
        model_id = "image-text/218"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture="qwen3vl",
            model_variant="2b-peft",
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        served_name = manager.resolve_and_register(
            model_id, metadata=metadata, api_key="key"
        )

        assert served_name.startswith("image-text-218-pkg1-")
        client.load_lora_adapter.assert_called_once()

    def test_misregistered_variant_accepted_when_adapter_config_matches_pool(
        self, monkeypatch, model_cache_dir
    ) -> None:
        _set_served_base(monkeypatch, variant="qwen3_5-2b")
        model_id = "image-text/223"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture="qwen3_5",
            model_variant="0.8b-peft",
        )
        downloads = _install_download_with_config(
            monkeypatch, base_model_name_or_path="qwen/qwen3_5-2b"
        )
        logger_mock = _install_logger_mock(monkeypatch)
        client = MagicMock()
        manager = AdapterManager(client=client)

        served_name = manager.resolve_and_register(
            model_id, metadata=metadata, api_key="key"
        )

        assert served_name.startswith("image-text-223-pkg1-")
        assert len(downloads) == 1
        client.load_lora_adapter.assert_called_once()
        assert manager.get_registration(served_name) is not None
        warnings = _rendered_warnings(logger_mock)
        drift_warnings = [m for m in warnings if "misregistered" in m]
        assert len(drift_warnings) == 1
        assert "image-text/223" in drift_warnings[0]
        assert "0.8b-peft" in drift_warnings[0]
        assert "qwen/qwen3_5-2b" in drift_warnings[0]
        _, load_kwargs = client.load_lora_adapter.call_args
        report = json.loads(
            (Path(load_kwargs["path"]) / "patch_report.json").read_text()
        )
        assert report["registry_variant"] == "0.8b-peft"
        assert report["base_model_name_or_path"] == "qwen/qwen3_5-2b"

    def test_registry_variant_contradicting_adapter_config_is_rejected_preflight(
        self, monkeypatch, model_cache_dir
    ) -> None:
        _set_served_base(monkeypatch, variant="qwen3_5-0.8b")
        model_id = "image-text/223"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture="qwen3_5",
            model_variant="0.8b-peft",
        )

        def _fake_download(target_dir: str, files_specs, verbose=True, **kwargs):
            write_adapter_package(
                target_dir=target_dir,
                config=build_adapter_config(base_model_name_or_path="qwen/qwen3_5-2b"),
            )

        monkeypatch.setattr(
            adapter_manager_module, "download_files_to_directory", _fake_download
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        with pytest.raises(AdapterNotServableError) as error:
            manager.resolve_and_register(model_id, metadata=metadata, api_key="key")
        message = str(error.value)
        assert "qwen/qwen3_5-2b" in message
        assert "qwen3_5-0.8b" in message
        assert "image-text/223" in message
        client.load_lora_adapter.assert_not_called()

    def test_vllm_5xx_on_load_is_surfaced_as_adapter_not_servable(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        client = MagicMock()
        client.load_lora_adapter.side_effect = VLLMHTTPError(
            message="vLLM sidecar returned HTTP 500 for POST /v1/load_lora_adapter",
            status_code=500,
            response_body=(
                "RuntimeError: The size of tensor a (1024) must match the "
                "size of tensor b (2048)"
            ),
        )
        manager = AdapterManager(client=client)

        with pytest.raises(AdapterNotServableError) as error:
            manager.resolve_and_register(
                model_id, metadata=build_metadata(model_id=model_id)
            )
        message = str(error.value)
        assert "some-workspace-some-project-1-pkg1-" in message
        assert model_id in message
        assert "size of tensor a (1024)" in message
        assert isinstance(error.value.__cause__, VLLMHTTPError)

    def test_connection_error_on_load_propagates_unchanged(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        client = MagicMock()
        client.load_lora_adapter.side_effect = VLLMConnectionError(
            "Could not reach vLLM sidecar"
        )
        manager = AdapterManager(client=client)

        with pytest.raises(VLLMConnectionError):
            manager.resolve_and_register(
                model_id, metadata=build_metadata(model_id=model_id)
            )

    def test_vllm_4xx_on_load_propagates_unchanged(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        model_id = "some-workspace/some-project/1"
        client = MagicMock()
        client.load_lora_adapter.side_effect = VLLMHTTPError(
            message="vLLM sidecar returned HTTP 400 for POST /v1/load_lora_adapter",
            status_code=400,
            response_body="invalid adapter",
        )
        manager = AdapterManager(client=client)

        with pytest.raises(VLLMHTTPError):
            manager.resolve_and_register(
                model_id, metadata=build_metadata(model_id=model_id)
            )

    def test_custom_served_base_variant_is_respected(self, monkeypatch) -> None:
        _set_served_base(monkeypatch, variant="qwen3_5-2b", name="qwen-base")
        provider = _install_provider(monkeypatch, {})
        manager = AdapterManager(client=MagicMock())

        served_name = manager.resolve_and_register("qwen3_5-2b")

        assert served_name == "qwen-base"
        provider.assert_not_called()


class TestLockScope:
    def test_registry_lock_is_not_held_during_download_patch_and_load(
        self, monkeypatch, model_cache_dir
    ) -> None:
        manager = AdapterManager(client=MagicMock())
        lock_states = []

        def _fake_download(target_dir: str, files_specs, verbose=True, **kwargs):
            lock_states.append(("download", manager._lock.locked()))
            write_adapter_package(target_dir=target_dir)

        def _fake_load(name: str, path: str) -> None:
            lock_states.append(("load", manager._lock.locked()))

        monkeypatch.setattr(
            adapter_manager_module, "download_files_to_directory", _fake_download
        )
        manager.client.load_lora_adapter.side_effect = _fake_load
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id)

        manager.resolve_and_register(model_id, metadata=metadata)
        manager.resolve_and_register(model_id, metadata=metadata)

        assert lock_states == [("download", False), ("load", False), ("load", False)]
        assert not manager._lock.locked()
