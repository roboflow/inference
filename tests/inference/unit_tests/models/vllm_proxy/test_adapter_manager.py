import os
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from inference.models.vllm_proxy import adapter_manager as adapter_manager_module
from inference.models.vllm_proxy.adapter_manager import (
    AdapterManager,
    normalize_base_variant,
)
from inference.models.vllm_proxy.errors import (
    AdapterNotServableError,
    NotServableOnVLLMError,
    VLLMConnectionError,
    VLLMHTTPError,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    ModelMetadata,
    ModelPackageMetadata,
)
from tests.inference.unit_tests.models.vllm_proxy.common import (
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
        model_variant=model_variant,
    )


@pytest.fixture
def fake_download(monkeypatch):
    """Replaces download_files_to_directory with a fixture-package writer."""
    calls = []

    def _fake_download(target_dir: str, files_specs, verbose=True, **kwargs):
        calls.append(target_dir)
        write_adapter_package(target_dir=target_dir)
        return {
            handle: os.path.join(target_dir, handle) for handle, _, _ in files_specs
        }

    monkeypatch.setattr(
        adapter_manager_module, "download_files_to_directory", _fake_download
    )
    return calls


@pytest.fixture
def model_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))
    return str(tmp_path / "cache")


def _install_provider(monkeypatch, metadata_by_model_id: dict) -> MagicMock:
    provider = MagicMock(
        side_effect=lambda model_id, provider, api_key=None, **kwargs: (
            metadata_by_model_id[model_id]
        )
    )
    monkeypatch.setattr(adapter_manager_module, "get_model_from_provider", provider)
    return provider


class TestNormalizeBaseVariant:
    @pytest.mark.parametrize(
        "architecture, variant, expected",
        [
            # qwen3_5 variants already carry the architecture prefix
            ("qwen3_5", "qwen3_5-0.8b", "qwen3_5-0.8b"),
            ("qwen3_5", "qwen3_5-0.8b-peft", "qwen3_5-0.8b"),
            # ... but bare variants are normalised the same way
            ("qwen3_5", "0.8b", "qwen3_5-0.8b"),
            ("qwen3_5", "0.8b-peft", "qwen3_5-0.8b"),
            # qwen3vl registry metadata reports bare variants
            ("qwen3vl", "2b", "qwen3vl-2b"),
            ("qwen3vl", "2b-peft", "qwen3vl-2b"),
            ("qwen3vl", "qwen3vl-2b-peft", "qwen3vl-2b"),
            # case-insensitive
            ("QWEN3VL", "2B-PEFT", "qwen3vl-2b"),
            # non-peft suffixes are preserved
            ("qwen3vl", "2b-instruct", "qwen3vl-2b-instruct"),
            # missing fields
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
    """Matrix for the base-variant verification of fine-tune metadata.

    The configured VLLM_SERVED_BASE_VARIANT must equal
    `<architecture>-<variant-with-peft-suffix-stripped>`.
    """

    @pytest.mark.parametrize(
        "architecture, variant, served_variant, accepted",
        [
            ("qwen3_5", "qwen3_5-0.8b", "qwen3_5-0.8b", True),
            ("qwen3_5", "0.8b-peft", "qwen3_5-0.8b", True),
            ("qwen3_5", "qwen3_5-0.8b-peft", "qwen3_5-0.8b", True),
            ("qwen3vl", "2b-peft", "qwen3vl-2b", True),
            ("qwen3vl", "2b", "qwen3vl-2b", True),
            ("qwen3vl", "2B-PEFT", "qwen3vl-2b", True),
            # family mismatch
            ("qwen3vl", "2b-peft", "qwen3_5-0.8b", False),
            ("qwen3_5", "0.8b-peft", "qwen3vl-2b", False),
            # size mismatch within the family
            ("qwen3_5", "qwen3_5-2b", "qwen3_5-0.8b", False),
            ("qwen3vl", "4b-peft", "qwen3vl-2b", False),
        ],
    )
    def test_fine_tune_variant_matching(
        self,
        monkeypatch,
        fake_download,
        model_cache_dir,
        architecture: str,
        variant: str,
        served_variant: str,
        accepted: bool,
    ) -> None:
        # given
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", served_variant)
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture=architecture,
            model_variant=variant,
        )
        _install_provider(monkeypatch, {model_id: metadata})
        manager = AdapterManager(client=MagicMock())

        # when / then
        if accepted:
            served_name = manager.resolve_and_register(model_id=model_id)
            assert served_name.startswith("some-workspace-some-project-1-")
        else:
            with pytest.raises(NotServableOnVLLMError):
                manager.resolve_and_register(model_id=model_id)


class TestResolveAndRegister:
    def test_base_model_id_returns_served_base_name_without_provider_call(
        self, monkeypatch
    ) -> None:
        # given
        provider = _install_provider(monkeypatch, {})
        manager = AdapterManager(client=MagicMock())

        # when
        served_name = manager.resolve_and_register(model_id="qwen3_5-0.8b")

        # then
        assert served_name == "qwen3_5-0.8b"
        provider.assert_not_called()

    def test_base_package_without_adapter_files_returns_served_base_name(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given
        metadata = build_metadata(model_id="qwen-base-alias", with_adapter_files=False)
        _install_provider(monkeypatch, {"qwen-base-alias": metadata})
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when
        served_name = manager.resolve_and_register(model_id="qwen-base-alias")

        # then
        assert served_name == "qwen3_5-0.8b"
        client.load_lora_adapter.assert_not_called()

    def test_fine_tune_is_downloaded_patched_and_registered(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given
        model_id = "some-workspace/some-project/1"
        _install_provider(monkeypatch, {model_id: build_metadata(model_id=model_id)})
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when
        served_name = manager.resolve_and_register(model_id=model_id, api_key="key")

        # then
        assert served_name.startswith("some-workspace-some-project-1-pkg1-")
        client.load_lora_adapter.assert_called_once()
        _, load_kwargs = client.load_lora_adapter.call_args
        assert load_kwargs["name"] == served_name
        assert os.path.isfile(
            os.path.join(load_kwargs["path"], "adapter_model.safetensors")
        )
        assert os.path.isfile(os.path.join(load_kwargs["path"], "patch_report.json"))

    def test_slug_includes_content_digest(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given - same model_id + package_id, different artifact hashes (the
        # package_id-is-not-unique-per-version quirk) must yield distinct
        # served names.
        model_id = "some-workspace/some-project/1"
        metadata_v1 = build_metadata(model_id=model_id, adapter_md5_prefix="aa")
        metadata_v2 = build_metadata(model_id=model_id, adapter_md5_prefix="bb")
        metadata_holder = {model_id: metadata_v1}
        _install_provider(monkeypatch, metadata_holder)
        manager = AdapterManager(client=MagicMock())

        # when
        served_v1 = manager.resolve_and_register(model_id=model_id)
        metadata_holder[model_id] = metadata_v2
        served_v2 = manager.resolve_and_register(model_id=model_id)

        # then
        assert served_v1 != served_v2

    def test_re_registration_is_idempotent(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given
        model_id = "some-workspace/some-project/1"
        _install_provider(monkeypatch, {model_id: build_metadata(model_id=model_id)})
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when
        served_first = manager.resolve_and_register(model_id=model_id)
        served_second = manager.resolve_and_register(model_id=model_id)

        # then
        assert served_first == served_second
        assert client.load_lora_adapter.call_count == 1
        assert len(fake_download) == 1

    def test_lru_eviction_unloads_oldest_adapter(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given
        monkeypatch.setenv("VLLM_MAX_REGISTERED_ADAPTERS", "2")
        metadata_by_model_id = {
            f"ws/project-{i}/1": build_metadata(
                model_id=f"ws/project-{i}/1",
                package_id=f"pkg{i}",
                adapter_md5_prefix=f"{i}{i}",
            )
            for i in range(3)
        }
        _install_provider(monkeypatch, metadata_by_model_id)
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when
        served_names = [
            manager.resolve_and_register(model_id=model_id)
            for model_id in metadata_by_model_id
        ]

        # then
        client.unload_lora_adapter.assert_called_once_with(name=served_names[0])
        assert manager.get_registration(served_names[0]) is None
        assert manager.get_registration(served_names[1]) is not None
        assert manager.get_registration(served_names[2]) is not None

    def test_wrong_base_variant_is_rejected(self, monkeypatch) -> None:
        # given
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id, model_variant="qwen3_5-2b")
        _install_provider(monkeypatch, {model_id: metadata})
        manager = AdapterManager(client=MagicMock())

        # when / then
        with pytest.raises(NotServableOnVLLMError):
            manager.resolve_and_register(model_id=model_id)

    def test_wrong_architecture_is_rejected(self, monkeypatch) -> None:
        # given
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id, model_architecture="qwen25vl")
        _install_provider(monkeypatch, {model_id: metadata})
        manager = AdapterManager(client=MagicMock())

        # when / then
        with pytest.raises(NotServableOnVLLMError):
            manager.resolve_and_register(model_id=model_id)

    def test_missing_hf_package_is_rejected(self, monkeypatch) -> None:
        # given
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id, backend=BackendType.ONNX)
        _install_provider(monkeypatch, {model_id: metadata})
        manager = AdapterManager(client=MagicMock())

        # when / then
        with pytest.raises(NotServableOnVLLMError):
            manager.resolve_and_register(model_id=model_id)

    def test_base_id_matching_served_base_name_short_circuits(
        self, monkeypatch
    ) -> None:
        # given - the qwen3vl base id ("qwen3vl-2b-instruct") differs from
        # the configured variant ("qwen3vl-2b"); it must short-circuit via
        # VLLM_SERVED_BASE_NAME without a provider call.
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", "qwen3vl-2b")
        monkeypatch.setenv("VLLM_SERVED_BASE_NAME", "qwen3vl-2b-instruct")
        provider = _install_provider(monkeypatch, {})
        manager = AdapterManager(client=MagicMock())

        # when
        served_name = manager.resolve_and_register(model_id="qwen3vl-2b-instruct")

        # then
        assert served_name == "qwen3vl-2b-instruct"
        provider.assert_not_called()

    def test_base_id_short_circuit_is_case_insensitive(self, monkeypatch) -> None:
        # given
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", "qwen3vl-2b")
        monkeypatch.setenv("VLLM_SERVED_BASE_NAME", "qwen3vl-2b-instruct")
        provider = _install_provider(monkeypatch, {})
        manager = AdapterManager(client=MagicMock())

        # when
        served_name = manager.resolve_and_register(model_id="QWEN3VL-2B")

        # then
        assert served_name == "qwen3vl-2b-instruct"
        provider.assert_not_called()

    def test_qwen3vl_fine_tune_is_registered_on_qwen3vl_pool(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given - registry metadata shape of the image-text/218 lab adapter:
        # architecture "qwen3vl", variant "2b-peft".
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", "qwen3vl-2b")
        monkeypatch.setenv("VLLM_SERVED_BASE_NAME", "qwen3vl-2b-instruct")
        model_id = "image-text/218"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture="qwen3vl",
            model_variant="2b-peft",
        )
        _install_provider(monkeypatch, {model_id: metadata})
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when
        served_name = manager.resolve_and_register(model_id=model_id, api_key="key")

        # then
        assert served_name.startswith("image-text-218-pkg1-")
        client.load_lora_adapter.assert_called_once()

    def test_registry_variant_contradicting_adapter_config_is_rejected_preflight(
        self, monkeypatch, model_cache_dir
    ) -> None:
        # given - exact reproduction of the image-text/223 incident
        # (2026-06-10, staging): registry metadata says modelVariant
        # "0.8b-peft" (so variant matching against the 0.8b pool passes),
        # but the adapter's own adapter_config.json declares
        # base_model_name_or_path "qwen/qwen3_5-2b".
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", "qwen3_5-0.8b")
        model_id = "image-text/223"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture="qwen3_5",
            model_variant="0.8b-peft",
        )
        _install_provider(monkeypatch, {model_id: metadata})

        def _fake_download(target_dir: str, files_specs, verbose=True, **kwargs):
            write_adapter_package(
                target_dir=target_dir,
                config=build_adapter_config(
                    base_model_name_or_path="qwen/qwen3_5-2b"
                ),
            )

        monkeypatch.setattr(
            adapter_manager_module, "download_files_to_directory", _fake_download
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when / then - rejected BEFORE any vLLM load attempt, naming both
        # the adapter's declared base and the served base
        with pytest.raises(AdapterNotServableError) as error:
            manager.resolve_and_register(model_id=model_id, api_key="key")
        message = str(error.value)
        assert "qwen/qwen3_5-2b" in message
        assert "qwen3_5-0.8b" in message
        assert "image-text/223" in message
        client.load_lora_adapter.assert_not_called()

    def test_vllm_5xx_on_load_is_surfaced_as_adapter_not_servable(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given - vLLM rejects the adapter deep inside set_lora (the opaque
        # tensor-shape 500 from the image-text/223 incident)
        model_id = "some-workspace/some-project/1"
        _install_provider(monkeypatch, {model_id: build_metadata(model_id=model_id)})
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

        # when / then
        with pytest.raises(AdapterNotServableError) as error:
            manager.resolve_and_register(model_id=model_id)
        message = str(error.value)
        assert "some-workspace-some-project-1-pkg1-" in message  # adapter slug
        assert model_id in message
        assert "size of tensor a (1024)" in message  # vLLM body excerpt
        assert isinstance(error.value.__cause__, VLLMHTTPError)

    def test_connection_error_on_load_propagates_unchanged(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given - sidecar down is an infra problem, not an adapter problem
        model_id = "some-workspace/some-project/1"
        _install_provider(monkeypatch, {model_id: build_metadata(model_id=model_id)})
        client = MagicMock()
        client.load_lora_adapter.side_effect = VLLMConnectionError(
            "Could not reach vLLM sidecar"
        )
        manager = AdapterManager(client=client)

        # when / then
        with pytest.raises(VLLMConnectionError):
            manager.resolve_and_register(model_id=model_id)

    def test_vllm_4xx_on_load_propagates_unchanged(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given - non-5xx HTTP errors keep their typed identity
        model_id = "some-workspace/some-project/1"
        _install_provider(monkeypatch, {model_id: build_metadata(model_id=model_id)})
        client = MagicMock()
        client.load_lora_adapter.side_effect = VLLMHTTPError(
            message="vLLM sidecar returned HTTP 400 for POST /v1/load_lora_adapter",
            status_code=400,
            response_body="invalid adapter",
        )
        manager = AdapterManager(client=client)

        # when / then
        with pytest.raises(VLLMHTTPError):
            manager.resolve_and_register(model_id=model_id)

    def test_custom_served_base_variant_is_respected(self, monkeypatch) -> None:
        # given
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", "qwen3_5-2b")
        monkeypatch.setenv("VLLM_SERVED_BASE_NAME", "qwen-base")
        provider = _install_provider(monkeypatch, {})
        manager = AdapterManager(client=MagicMock())

        # when
        served_name = manager.resolve_and_register(model_id="qwen3_5-2b")

        # then
        assert served_name == "qwen-base"
        provider.assert_not_called()
