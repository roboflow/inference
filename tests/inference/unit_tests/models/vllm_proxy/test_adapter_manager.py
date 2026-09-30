import json
import os
import shutil
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
    monkeypatch.setattr(adapter_manager_module, "logger", logger_mock)
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
    """Matrix for the ADVISORY base-variant check of fine-tune metadata.

    The configured VLLM_SERVED_BASE_VARIANT equals
    `<architecture>-<variant-with-peft-suffix-stripped>`. Registry
    `modelVariant` is sometimes misregistered (incident 2026-06-10:
    `image-text/223`), so a mismatch no longer rejects pre-download - it
    logs a warning and defers to the adapter's own `adapter_config.json`
    (`cross_check_base_model` in `patch_adapter`), which is authoritative.
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
        # given
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", served_variant)
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture=architecture,
            model_variant=variant,
        )
        _install_provider(monkeypatch, {model_id: metadata})
        logger_mock = _install_logger_mock(monkeypatch)
        manager = AdapterManager(client=MagicMock())

        # when
        served_name = manager.resolve_and_register(model_id=model_id)

        # then
        assert served_name.startswith("some-workspace-some-project-1-")
        logger_mock.warning.assert_not_called()

    @pytest.mark.parametrize(
        "architecture, variant, served_variant",
        [
            # family mismatch
            ("qwen3vl", "2b-peft", "qwen3_5-0.8b"),
            ("qwen3_5", "0.8b-peft", "qwen3vl-2b"),
            # size mismatch within the family
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
        # given - the registry variant contradicts this pool, but the
        # adapter's own config declares this pool's base. adapter_config is
        # authoritative, so the adapter must be ACCEPTED (these rows were
        # rejected by the pre-download variant gate before the registry was
        # demoted to advisory).
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", served_variant)
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture=architecture,
            model_variant=variant,
        )
        _install_provider(monkeypatch, {model_id: metadata})
        downloads = _install_download_with_config(
            monkeypatch, base_model_name_or_path=f"qwen/{served_variant}"
        )
        logger_mock = _install_logger_mock(monkeypatch)
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when
        served_name = manager.resolve_and_register(model_id=model_id)

        # then - downloaded, accepted, registered; both warnings emitted
        # (variant-gate deferral + misregistration drift audit)
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

    def test_re_registration_skips_download_but_always_recalls_vllm_load(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given - with NUM_WORKERS>1 the per-process map can be stale
        # relative to the shared vLLM engine, so a slug already in the map
        # must STILL re-issue the idempotent load_lora_adapter call, while
        # the expensive download/patch work stays skipped.
        model_id = "some-workspace/some-project/1"
        _install_provider(monkeypatch, {model_id: build_metadata(model_id=model_id)})
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when
        served_first = manager.resolve_and_register(model_id=model_id)
        served_second = manager.resolve_and_register(model_id=model_id)

        # then
        assert served_first == served_second
        assert client.load_lora_adapter.call_count == 2
        for _, load_kwargs in client.load_lora_adapter.call_args_list:
            assert load_kwargs["name"] == served_first
        assert len(fake_download) == 1

    def test_recorded_slug_with_missing_patched_dir_redoes_download_and_patch(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given - the map records the slug but the patched dir vanished from
        # disk (external GC): the download/patch fast-path must not be taken.
        model_id = "some-workspace/some-project/1"
        _install_provider(monkeypatch, {model_id: build_metadata(model_id=model_id)})
        client = MagicMock()
        manager = AdapterManager(client=client)
        served_name = manager.resolve_and_register(model_id=model_id)
        shutil.rmtree(manager.get_registration(served_name).patched_dir)

        # when
        served_again = manager.resolve_and_register(model_id=model_id)

        # then
        assert served_again == served_name
        assert len(fake_download) == 2
        assert os.path.isdir(manager.get_registration(served_name).patched_dir)

    def test_partial_patched_dir_redoes_download_and_patch(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given - a prior worker died after creating the patched dir but before
        # all files were published. The manager must not hand that path to vLLM.
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id)
        _install_provider(monkeypatch, {model_id: metadata})
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
        patched_dir = os.path.join(model_cache_dir, "vllm-adapters", slug, "patched")
        os.makedirs(patched_dir, exist_ok=True)
        with open(os.path.join(patched_dir, "adapter_config.json"), "w") as f:
            json.dump({"partial": True}, f)

        # when
        served_name = manager.resolve_and_register(model_id=model_id)

        # then
        assert served_name == slug
        assert len(fake_download) == 1
        _, load_kwargs = manager.client.load_lora_adapter.call_args
        assert os.path.isfile(
            os.path.join(load_kwargs["path"], "adapter_model.safetensors")
        )
        assert os.path.isfile(os.path.join(load_kwargs["path"], "patch_report.json"))

    def test_runtime_svd_policy_is_rejected_before_download(
        self, monkeypatch, model_cache_dir
    ) -> None:
        # given - adapter_manager intentionally prunes base/ artifacts, so the
        # request-path cannot satisfy patch_adapter(policy="svd", base_dir=...).
        monkeypatch.setenv("VLLM_DORA_POLICY", "svd")
        model_id = "some-workspace/some-project/1"
        _install_provider(monkeypatch, {model_id: build_metadata(model_id=model_id)})
        download = MagicMock()
        monkeypatch.setattr(
            adapter_manager_module, "download_files_to_directory", download
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when / then
        with pytest.raises(NotServableOnVLLMError) as error:
            manager.resolve_and_register(model_id=model_id)
        assert "VLLM_DORA_POLICY=svd" in str(error.value)
        assert "base/ weights" in str(error.value)
        download.assert_not_called()
        client.load_lora_adapter.assert_not_called()

    def test_overflow_past_max_registered_warns_and_never_unloads(
        self, monkeypatch, fake_download, model_cache_dir
    ) -> None:
        # given - VLLM_MAX_REGISTERED_ADAPTERS is a warn-only threshold:
        # all gunicorn workers share ONE vLLM engine, so an automatic unload
        # driven by one worker's bookkeeping would yank adapters the other
        # workers still serve. vLLM's own --max-cpu-loras LRU bounds memory.
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
        logger_mock = _install_logger_mock(monkeypatch)
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when
        served_names = [
            manager.resolve_and_register(model_id=model_id)
            for model_id in metadata_by_model_id
        ]

        # then - no unload, all registrations retained, WARN emitted
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
        # given
        model_id = "some-workspace/some-project/1"
        _install_provider(monkeypatch, {model_id: build_metadata(model_id=model_id)})
        client = MagicMock()
        manager = AdapterManager(client=client)
        served_name = manager.resolve_and_register(model_id=model_id)

        # when
        manager.invalidate(served_name=served_name)

        # then - dropped from the map; re-resolution registers again (files
        # already on disk are re-downloaded by the fake, re-patched, and the
        # vLLM load call re-issued) and never unloads
        assert manager.get_registration(served_name) is None
        assert manager.resolve_and_register(model_id=model_id) == served_name
        assert manager.get_registration(served_name) is not None
        assert client.load_lora_adapter.call_count == 2
        client.unload_lora_adapter.assert_not_called()

    def test_wrong_base_variant_is_rejected_by_adapter_config_cross_check(
        self, monkeypatch, model_cache_dir
    ) -> None:
        # given - registry variant qwen3_5-2b at the (default) qwen3_5-0.8b
        # pool, with an adapter_config that agrees (a genuine 2b fine-tune).
        # The pre-download variant gate used to raise here; the registry is
        # now advisory, so the download proceeds and the authoritative
        # adapter_config cross-check rejects, naming both bases.
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id, model_variant="qwen3_5-2b")
        _install_provider(monkeypatch, {model_id: metadata})
        downloads = _install_download_with_config(
            monkeypatch, base_model_name_or_path="qwen/qwen3_5-2b"
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when / then
        with pytest.raises(AdapterNotServableError) as error:
            manager.resolve_and_register(model_id=model_id)
        assert len(downloads) == 1  # the variant gate did NOT block
        message = str(error.value)
        assert "qwen/qwen3_5-2b" in message
        assert "qwen3_5-0.8b" in message
        client.load_lora_adapter.assert_not_called()

    @pytest.mark.parametrize("architecture", ["qwen25vl", "florence2"])
    def test_wrong_architecture_is_rejected_pre_download(
        self, monkeypatch, architecture: str
    ) -> None:
        # given - modelArchitecture (unlike modelVariant) has not been
        # observed misregistered, so it stays a hard pre-download gate
        # preventing florence/qwen2.5 artifact downloads.
        model_id = "some-workspace/some-project/1"
        metadata = build_metadata(model_id=model_id, model_architecture=architecture)
        _install_provider(monkeypatch, {model_id: metadata})
        download = MagicMock()
        monkeypatch.setattr(
            adapter_manager_module, "download_files_to_directory", download
        )
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when / then
        with pytest.raises(NotServableOnVLLMError):
            manager.resolve_and_register(model_id=model_id)
        download.assert_not_called()
        client.load_lora_adapter.assert_not_called()

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

    def test_misregistered_variant_accepted_when_adapter_config_matches_pool(
        self, monkeypatch, model_cache_dir
    ) -> None:
        # given - image-text/223 arriving at the pool its FILE belongs to:
        # registry metadata misregisters modelVariant as "0.8b-peft", but
        # the adapter_config.json declares base "qwen/qwen3_5-2b" and this
        # pool serves qwen3_5-2b. Once the producer routes by file, this
        # pool must ACCEPT the adapter despite the registry variant, and
        # emit the drift-audit warning naming both values.
        monkeypatch.setenv("VLLM_SERVED_BASE_VARIANT", "qwen3_5-2b")
        model_id = "image-text/223"
        metadata = build_metadata(
            model_id=model_id,
            model_architecture="qwen3_5",
            model_variant="0.8b-peft",
        )
        _install_provider(monkeypatch, {model_id: metadata})
        downloads = _install_download_with_config(
            monkeypatch, base_model_name_or_path="qwen/qwen3_5-2b"
        )
        logger_mock = _install_logger_mock(monkeypatch)
        client = MagicMock()
        manager = AdapterManager(client=client)

        # when
        served_name = manager.resolve_and_register(model_id=model_id, api_key="key")

        # then - accepted and registered
        assert served_name.startswith("image-text-223-pkg1-")
        assert len(downloads) == 1
        client.load_lora_adapter.assert_called_once()
        assert manager.get_registration(served_name) is not None
        # the drift-audit warning names the model and both values
        warnings = _rendered_warnings(logger_mock)
        drift_warnings = [m for m in warnings if "misregistered" in m]
        assert len(drift_warnings) == 1
        assert "image-text/223" in drift_warnings[0]
        assert "0.8b-peft" in drift_warnings[0]
        assert "qwen/qwen3_5-2b" in drift_warnings[0]
        # both values are recorded in the patch report
        _, load_kwargs = client.load_lora_adapter.call_args
        with open(os.path.join(load_kwargs["path"], "patch_report.json")) as f:
            report = json.load(f)
        assert report["registry_variant"] == "0.8b-peft"
        assert report["base_model_name_or_path"] == "qwen/qwen3_5-2b"

    def test_registry_variant_contradicting_adapter_config_is_rejected_preflight(
        self, monkeypatch, model_cache_dir
    ) -> None:
        # given - exact reproduction of the image-text/223 incident
        # (2026-06-10, staging) AT THE WRONG POOL: registry metadata says
        # modelVariant "0.8b-peft" (so the advisory variant check against
        # the 0.8b pool passes), but the adapter's own adapter_config.json
        # declares base_model_name_or_path "qwen/qwen3_5-2b" - the
        # authoritative cross-check must reject before any vLLM load.
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
                config=build_adapter_config(base_model_name_or_path="qwen/qwen3_5-2b"),
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
