import asyncio
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from inference.core.exceptions import ModelDeploymentNotSupportedError
from inference.core.managers import mmp_translation as translation
from inference.core.managers.mmp_adapter import ModelManagerAdapter


class FakeStatClient:
    load_wait_s = 1.0
    infer_timeout_s = 1.0
    n_slots = 4

    def __init__(self):
        self.loaded = []
        self.unloaded = []
        self.tasks = {"embed_images": {}, "embed_text": {}, "compare": {}}
        self.load_result = ("ok",)

    async def start(self):
        pass

    async def shutdown(self):
        pass

    async def load(self, model_id, api_key=""):
        self.loaded.append(model_id)
        return self.load_result

    async def unload(self, model_id):
        self.unloaded.append(model_id)
        return ("ok",)

    async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
        return ("model_ready",)

    async def interface(self, model_id):
        return {"model_id": model_id, "tasks": self.tasks}

    async def stats(self):
        entry = {
            "class_names": None,
            "key_points_classes": None,
            "model_class_name": "ClipOnnx",
        }
        return {"mmp_models": {m: dict(entry) for m in self.loaded}}


class FakeLegacy:
    def init_pingback(self):
        pass

    def record_request_metadata(self, **kwargs):
        pass


@pytest.fixture
def platform_stat_spy(monkeypatch):
    root = types.ModuleType("inference_server")
    framework = types.ModuleType("inference_server.framework")
    entities = types.ModuleType("inference_server.framework.entities")
    model_stat = types.ModuleType("inference_server.framework.model_stat")

    class CommonRequestParams:
        def __init__(self, model_id, api_key):
            self.model_id = model_id
            self.api_key = api_key

    entities.CommonRequestParams = CommonRequestParams
    spy = SimpleNamespace(calls=[], result=("object-detection", "infer"))

    async def stat_model_while_checking_auth(common_params):
        spy.calls.append((common_params.model_id, common_params.api_key))
        return spy.result

    model_stat.stat_model_while_checking_auth = stat_model_while_checking_auth
    for name, module in (
        ("inference_server", root),
        ("inference_server.framework", framework),
        ("inference_server.framework.entities", entities),
        ("inference_server.framework.model_stat", model_stat),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return spy


@pytest.fixture
def running_adapter():
    adapter = ModelManagerAdapter(
        legacy_stack=FakeLegacy(), mmp_client=FakeStatClient()
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(adapter.start(), loop).result(timeout=5)
    yield adapter
    asyncio.run_coroutine_threadsafe(adapter.shutdown(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


class TestGenericCoreIdResolution:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("clip/ViT-B-16", ("embedding", "embed_images")),
            ("clip", ("embedding", "embed_images")),
            ("perception_encoder/PE-Core-L14-336", ("embedding", "embed_images")),
            ("sam/vit_h", ("interactive-instance-segmentation", "embed")),
            ("sam2/hiera_large", ("interactive-instance-segmentation", "embed")),
            ("doctr/default", ("structured-ocr", "infer")),
            ("easy_ocr/english_g2", ("structured-ocr", "infer")),
            ("trocr/trocr-base-printed", ("text-only-ocr", "infer")),
        ],
    )
    def test_generic_core_id_resolves_without_platform_call(
        self, platform_stat_spy, model_id, expected
    ):
        result = asyncio.run(translation.stat_model(model_id=model_id, api_key="key"))
        assert result == expected
        assert platform_stat_spy.calls == []

    @pytest.mark.parametrize(
        "model_id",
        [
            "ws/1",
            "yolo_world/l",
            "sam3/sam3_final",
            "sam3/sam3_interactive",
            "gaze/l2cs",
            "pp_ocr/small",
            "paligemma-pretrains/1",
        ],
    )
    def test_non_bridged_id_falls_through_to_platform_stat(
        self, platform_stat_spy, model_id
    ):
        result = asyncio.run(translation.stat_model(model_id=model_id, api_key="key"))
        assert result == ("object-detection", "infer")
        assert platform_stat_spy.calls == [(model_id, "key")]


class TestCoreModelIdCanonicalization:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            (
                "perception_encoder/PE-Core-L14-336",
                "perception-encoder/PE-Core-L14-336",
            ),
            ("perception_encoder", "perception-encoder"),
            ("clip/ViT-B-16", "clip/ViT-B-16"),
            ("sam/vit_h", "sam/vit_h"),
            ("ws/1", "ws/1"),
        ],
    )
    def test_canonical_mmp_model_id(self, model_id, expected):
        assert translation.canonical_mmp_model_id(model_id) == expected

    def test_canonical_pe_id_resolves_without_platform_call(self, platform_stat_spy):
        result = asyncio.run(
            translation.stat_model(
                model_id="perception-encoder/PE-Core-L14-336", api_key="key"
            )
        )
        assert result == ("embedding", "embed_images")
        assert platform_stat_spy.calls == []

    def test_interface_failure_reports_legacy_id(
        self, running_adapter, platform_stat_spy
    ):
        interface_calls = []

        async def failing_interface(model_id):
            interface_calls.append(model_id)
            raise RuntimeError(f"model '{model_id}' is not loaded")

        running_adapter._client.interface = failing_interface
        with pytest.raises(RuntimeError) as exc_info:
            running_adapter.add_model(
                "perception_encoder/PE-Core-L14-336", api_key="key"
            )
        assert interface_calls == ["perception-encoder/PE-Core-L14-336"]
        assert "perception_encoder/PE-Core-L14-336" in str(exc_info.value)
        assert "perception-encoder" not in str(exc_info.value)

    def test_adapter_sends_platform_id_to_mmp_for_legacy_pe_id(
        self, running_adapter, platform_stat_spy
    ):
        running_adapter.add_model("perception_encoder/PE-Core-L14-336", api_key="key")
        assert running_adapter._client.loaded == ["perception-encoder/PE-Core-L14-336"]
        assert (
            running_adapter.get_task_type("perception_encoder/PE-Core-L14-336")
            == "embedding"
        )
        assert [d.model_id for d in running_adapter.describe_models()] == [
            "perception_encoder/PE-Core-L14-336"
        ]
        assert platform_stat_spy.calls == []
        running_adapter.remove("perception_encoder/PE-Core-L14-336")
        assert running_adapter._client.unloaded == [
            "perception-encoder/PE-Core-L14-336"
        ]


class TestPlatformTaskTypeAliases:
    def test_alias_normalized_to_canonical_task_type(
        self, platform_stat_spy, monkeypatch
    ):
        monkeypatch.setitem(
            translation._PLATFORM_TASK_TYPE_ALIASES,
            "platform-od-alias",
            "object-detection",
        )
        platform_stat_spy.result = ("platform-od-alias", "infer")
        result = asyncio.run(translation.stat_model(model_id="ws/1", api_key="key"))
        assert result == ("object-detection", "infer")

    def test_alias_normalization_recomputes_default_action(
        self, platform_stat_spy, monkeypatch
    ):
        monkeypatch.setitem(
            translation._PLATFORM_TASK_TYPE_ALIASES,
            "platform-embedding-alias",
            "embedding",
        )
        platform_stat_spy.result = ("platform-embedding-alias", "infer")
        result = asyncio.run(translation.stat_model(model_id="ws/1", api_key="key"))
        assert result == ("embedding", "embed_images")

    def test_unknown_task_type_passes_through_verbatim(self, platform_stat_spy):
        platform_stat_spy.result = ("gaze-detection", "infer")
        result = asyncio.run(translation.stat_model(model_id="l2cs/1", api_key="key"))
        assert result == ("gaze-detection", "infer")


class TestAdapterStatIntegration:
    def test_generic_clip_id_loads_without_platform_stat(
        self, running_adapter, platform_stat_spy
    ):
        running_adapter.add_model("clip/ViT-B-16", api_key="key")
        assert running_adapter._client.loaded == ["clip/ViT-B-16"]
        assert running_adapter.get_task_type("clip/ViT-B-16") == "embedding"
        assert platform_stat_spy.calls == []

    def test_unknown_platform_task_type_still_refused(
        self, running_adapter, platform_stat_spy
    ):
        platform_stat_spy.result = ("gaze-detection", "infer")
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("l2cs/1", api_key="key")
        assert running_adapter._client.loaded == []
        assert platform_stat_spy.calls == [("l2cs/1", "key")]
