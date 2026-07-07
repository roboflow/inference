import asyncio
import threading

import pytest

from inference.core.exceptions import (
    InferenceModelNotFound,
    ModelDeploymentNotSupportedError,
)
from inference.core.managers.mmp_adapter import ModelManagerAdapter


class FakeClient:
    load_wait_s = 1.0
    infer_timeout_s = 1.0

    def __init__(self):
        self.started = False
        self.unloaded = []

    async def start(self):
        self.started = True

    async def shutdown(self):
        self.started = False

    async def unload(self, model_id):
        self.unloaded.append(model_id)
        return ("ok",)


class FakeLegacy:
    def __init__(self):
        self.pingback = None
        self.metadata_calls = []

    def init_pingback(self):
        self.num_errors = 0

    def record_request_metadata(self, **kwargs):
        self.metadata_calls.append(kwargs)


def make_adapter():
    return ModelManagerAdapter(legacy_stack=FakeLegacy(), mmp_client=FakeClient())


@pytest.fixture
def running_adapter():
    adapter = make_adapter()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(adapter.start(), loop).result(timeout=5)
    yield adapter
    asyncio.run_coroutine_threadsafe(adapter.shutdown(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


class TestUnsupportedModelOps:
    def test_add_model_raises(self):
        with pytest.raises(ModelDeploymentNotSupportedError):
            make_adapter().add_model("some/1", api_key="key")

    def test_infer_from_request_sync_raises(self):
        with pytest.raises(ModelDeploymentNotSupportedError):
            make_adapter().infer_from_request_sync("some/1", request=None)

    def test_infer_from_request_raises(self):
        with pytest.raises(ModelDeploymentNotSupportedError):
            asyncio.run(make_adapter().infer_from_request("some/1", request=None))

    def test_get_task_type_raises(self):
        with pytest.raises(ModelDeploymentNotSupportedError):
            make_adapter().get_task_type("some/1")

    def test_get_class_names_raises(self):
        with pytest.raises(ModelDeploymentNotSupportedError):
            make_adapter().get_class_names("some/1")

    def test_lower_level_ops_raise(self):
        adapter = make_adapter()
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.predict("some/1")
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.model_infer_sync("some/1", request=None)
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.preprocess("some/1", request=None)
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.postprocess("some/1")
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.make_response("some/1")


class TestContainerProtocol:
    def test_empty_state(self):
        adapter = make_adapter()
        assert len(adapter) == 0
        assert "some/1" not in adapter
        assert list(adapter.keys()) == []
        assert adapter.models() == {}
        assert adapter.describe_models() == []

    def test_route_state_visible(self):
        adapter = make_adapter()
        adapter._routes["some/1"] = {"task_type": "object-detection"}
        assert len(adapter) == 1
        assert "some/1" in adapter
        assert list(adapter.keys()) == ["some/1"]
        assert list(adapter.models()) == ["some/1"]

    def test_getitem_returns_inert_stub(self):
        adapter = make_adapter()
        adapter._routes["some/1"] = {"task_type": "object-detection"}
        stub = adapter["some/1"]
        assert stub.model_id == "some/1"
        assert not hasattr(stub, "flush")
        assert getattr(stub, "_pipeline_depth", None) is None

    def test_getitem_unknown_raises_model_not_found(self):
        with pytest.raises(InferenceModelNotFound):
            make_adapter()["missing/1"]


class TestLegacyDelegation:
    def test_init_pingback_and_attributes_delegate(self):
        adapter = make_adapter()
        adapter.init_pingback()
        assert adapter.num_errors == 0
        assert adapter.pingback is None

    def test_record_request_metadata_delegates(self):
        adapter = make_adapter()
        adapter.record_request_metadata(model_id="some/1")
        assert adapter._legacy.metadata_calls == [{"model_id": "some/1"}]

    def test_pin_model_is_noop(self):
        make_adapter().pin_model("some/1")


class TestSyncBridge:
    def test_round_trips_result_from_worker_thread(self, running_adapter):
        async def coro():
            return 42

        assert running_adapter._run_sync(coro()) == 42

    def test_raises_when_called_on_adapter_loop(self, running_adapter):
        async def on_loop():
            running_adapter._run_sync(asyncio.sleep(0))

        future = asyncio.run_coroutine_threadsafe(on_loop(), running_adapter._loop)
        with pytest.raises(RuntimeError, match="deadlock"):
            future.result(timeout=5)

    def test_raises_before_start(self):
        with pytest.raises(RuntimeError, match="before start"):
            make_adapter()._run_sync(asyncio.sleep(0))


class TestRemoveAndClear:
    def test_remove_unloads_and_drops_route(self, running_adapter):
        running_adapter._routes["some/1"] = {"task_type": "object-detection"}
        running_adapter.remove("some/1")
        assert running_adapter._client.unloaded == ["some/1"]
        assert "some/1" not in running_adapter

    def test_remove_unknown_is_noop(self, running_adapter):
        running_adapter.remove("missing/1")
        assert running_adapter._client.unloaded == []

    def test_clear_unloads_all(self, running_adapter):
        running_adapter._routes["a/1"] = {}
        running_adapter._routes["b/2"] = {}
        running_adapter.clear()
        assert sorted(running_adapter._client.unloaded) == ["a/1", "b/2"]
        assert len(running_adapter) == 0
