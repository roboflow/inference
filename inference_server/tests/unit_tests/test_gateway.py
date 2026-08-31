from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_server.gateway import (
    ModelManagerGateway,
    routed_model_id,
    routing_key,
)


def _fake_manager(process_return=None):
    mgr = MagicMock()
    mgr.process_async = AsyncMock(return_value=process_return)
    mgr.load = MagicMock()
    mgr.unload = MagicMock()
    mgr.stats = MagicMock(return_value={"models": []})
    mgr.n_slots = 32
    # No dedicated pool on the double — the gateway falls back to the loop's
    # default executor.
    mgr.executor = None
    return mgr


@pytest.mark.asyncio
async def test_infer_forwards_task_image_and_params_to_manager():
    mgr = _fake_manager(process_return={"detections": []})
    wrapper = ModelManagerGateway(mgr)
    image = b"\xff\xd8\xff"
    result = await wrapper.infer(
        model_id="acme/1",
        image=image,
        task="prompt",
        instance="",
        params={"confidence": 0.5, "prompt": "hi"},
    )
    assert result == {"detections": []}
    mgr.process_async.assert_awaited_once()
    args, kwargs = mgr.process_async.await_args
    assert args == ("acme/1",)
    assert kwargs["task"] == "prompt"
    assert kwargs["images"] == image
    assert kwargs["confidence"] == 0.5
    assert kwargs["prompt"] == "hi"


@pytest.mark.asyncio
async def test_infer_with_no_params_still_includes_images_kwarg():
    mgr = _fake_manager(process_return="ok")
    wrapper = ModelManagerGateway(mgr)
    await wrapper.infer(model_id="m", image=b"x", task=None, params=None)
    kwargs = mgr.process_async.await_args.kwargs
    assert kwargs["images"] == b"x"
    assert kwargs["task"] is None


@pytest.mark.asyncio
async def test_infer_forwards_numpy_image_to_manager_without_copying():
    import numpy as np

    mgr = _fake_manager(process_return="ok")
    wrapper = ModelManagerGateway(mgr)
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    await wrapper.infer(model_id="m", image=image, task="infer")

    kwargs = mgr.process_async.await_args.kwargs
    assert kwargs["images"] is image


@pytest.mark.asyncio
async def test_ensure_loaded_returns_model_ready_when_loaded():
    mgr = _fake_manager()
    mgr.stats = MagicMock(return_value={"models": [{"model_id": "acme/1"}]})
    wrapper = ModelManagerGateway(mgr)
    status = await wrapper.ensure_loaded("acme/1")
    assert status[0] == "model_ready"


@pytest.mark.asyncio
async def test_stats_rekeys_models_list_into_dict():
    mgr = _fake_manager()
    mgr.stats = MagicMock(
        return_value={
            "a": 1,
            "models": [{"model_id": "acme/1", "tasks": {"infer": {}}}],
        }
    )
    wrapper = ModelManagerGateway(mgr)
    out = await wrapper.stats()
    assert out["a"] == 1
    assert out["models"] == {
        "acme/1": {"model_id": "acme/1", "tasks": {"infer": {}}}
    }


@pytest.mark.asyncio
async def test_stats_includes_n_slots_when_manager_reports_it():
    mgr = _fake_manager()
    mgr.n_slots = 32
    wrapper = ModelManagerGateway(mgr)
    out = await wrapper.stats()
    assert out["n_slots"] == 32


@pytest.mark.asyncio
async def test_stats_omits_n_slots_when_manager_has_none():
    class _PlainManager:
        def stats(self):
            return {"models": []}

    wrapper = ModelManagerGateway(_PlainManager())
    out = await wrapper.stats()
    assert "n_slots" not in out


@pytest.mark.asyncio
async def test_interface_raises_runtime_error_when_model_not_loaded():
    mgr = _fake_manager()
    wrapper = ModelManagerGateway(mgr)
    with pytest.raises(RuntimeError, match="not loaded"):
        await wrapper.interface("ghost")


@pytest.mark.asyncio
async def test_interface_returns_tasks_for_loaded_model():
    mgr = _fake_manager()
    mgr.stats = MagicMock(
        return_value={"models": [{"model_id": "acme/1", "tasks": {"infer": {}}}]}
    )
    wrapper = ModelManagerGateway(mgr)
    info = await wrapper.interface("acme/1")
    assert info["model_id"] == "acme/1"
    assert info["tasks"] == {"infer": {}}


@pytest.mark.asyncio
async def test_infer_requests_raw_prediction_from_manager():
    """Direct gateway must hand L1 the RAW prediction (serialize=False) — the
    registry-typed dict broke L1 serializers that expect .xyxy etc."""
    mgr = _fake_manager(process_return={"raw": 1})
    wrapper = ModelManagerGateway(mgr)
    await wrapper.infer(model_id="m", image=b"x")
    kwargs = mgr.process_async.await_args.kwargs
    assert kwargs["serialize"] is False


@pytest.mark.asyncio
async def test_concurrent_ensure_loaded_loads_once():
    import asyncio
    import threading
    import time

    class _SlowManager:
        def __init__(self):
            self.loaded = set()
            self.load_calls = 0
            self._lock = threading.Lock()

        def __contains__(self, model_id):
            return model_id in self.loaded

        def load(self, model_id, api_key, device=None):
            with self._lock:
                self.load_calls += 1
            time.sleep(0.05)
            self.loaded.add(model_id)

    mgr = _SlowManager()
    wrapper = ModelManagerGateway(mgr)
    results = await asyncio.gather(
        wrapper.ensure_loaded("m"), wrapper.ensure_loaded("m")
    )
    assert [r[0] for r in results] == ["model_ready", "model_ready"]
    assert mgr.load_calls == 1


@pytest.mark.asyncio
async def test_load_deadline_raises_timeout_like_mmp_client():
    import asyncio
    import threading

    release = threading.Event()

    class _BlockedManager:
        def __init__(self):
            self.loaded = set()

        def __contains__(self, model_id):
            return model_id in self.loaded

        def load(self, model_id, api_key, **kwargs):
            release.wait(timeout=5)
            self.loaded.add(model_id)

    mgr = _BlockedManager()
    wrapper = ModelManagerGateway(mgr)
    with pytest.raises(asyncio.TimeoutError):
        await wrapper.load("m", "key", timeout_s=0.05)
    release.set()
    for _ in range(100):
        if "m" in mgr.loaded:
            break
        await asyncio.sleep(0.02)
    assert await wrapper.load("m", "key", timeout_s=0.05) == ("ok",)


@pytest.mark.asyncio
async def test_load_already_loaded_returns_ok_without_manager_call():
    class _Manager:
        def __init__(self):
            self.load_calls = 0

        def __contains__(self, model_id):
            return True

        def load(self, model_id, api_key, **kwargs):
            self.load_calls += 1

    mgr = _Manager()
    wrapper = ModelManagerGateway(mgr)
    assert await wrapper.load("m", "key") == ("ok",)
    assert mgr.load_calls == 0


@pytest.mark.asyncio
async def test_load_failure_maps_to_load_failed_code():
    class _Manager:
        def __contains__(self, model_id):
            return False

        def load(self, model_id, api_key, **kwargs):
            raise RuntimeError("weights download failed")

    wrapper = ModelManagerGateway(_Manager())
    assert await wrapper.load("m", "key") == ("error", 5)
    assert (await wrapper.ensure_loaded("m"))[:2] == ("error", 5)


@pytest.mark.asyncio
async def test_unload_missing_model_maps_to_not_loaded_code():
    class _Manager:
        def unload(self, model_id):
            raise KeyError(model_id)

    wrapper = ModelManagerGateway(_Manager())
    assert await wrapper.unload("ghost") == ("error", 6)


@pytest.mark.asyncio
async def test_load_kwargs_forwarded_to_manager_load():
    class _Manager:
        def __init__(self):
            self.load_kwargs = None

        def __contains__(self, model_id):
            return False

        def load(self, model_id, api_key, **kwargs):
            self.load_kwargs = kwargs

    mgr = _Manager()
    wrapper = ModelManagerGateway(mgr, load_kwargs={"backend": "subprocess"})
    assert await wrapper.load("m", "key") == ("ok",)
    assert mgr.load_kwargs["backend"] == "subprocess"

    mgr_default = _Manager()
    wrapper_default = ModelManagerGateway(mgr_default)
    await wrapper_default.load("m", "key")
    assert mgr_default.load_kwargs == {}


def test_budget_attrs_default_and_override():
    from inference_server import configuration

    mgr = _fake_manager()
    wrapper = ModelManagerGateway(mgr)
    assert wrapper.load_wait_s == configuration.LOAD_WAIT_S
    assert wrapper.infer_timeout_s == configuration.INFER_TIMEOUT_S
    assert wrapper.n_slots == 32

    wrapper_override = ModelManagerGateway(mgr, load_wait_s=600.0, infer_timeout_s=300.0)
    assert wrapper_override.load_wait_s == 600.0
    assert wrapper_override.infer_timeout_s == 300.0


@pytest.mark.asyncio
async def test_timed_out_load_shares_future_with_next_call():
    import threading

    release = threading.Event()

    class _Manager:
        def __init__(self):
            self.load_calls = 0
            self.loaded = set()

        def __contains__(self, model_id):
            return model_id in self.loaded

        def load(self, model_id, api_key, **kwargs):
            self.load_calls += 1
            release.wait(timeout=5)
            self.loaded.add(model_id)

    import asyncio

    mgr = _Manager()
    wrapper = ModelManagerGateway(mgr)
    with pytest.raises(asyncio.TimeoutError):
        await wrapper.load("m", "key", timeout_s=0.05)
    release.set()
    assert await wrapper.load("m", "key") == ("ok",)
    assert mgr.load_calls == 1


@pytest.mark.asyncio
async def test_ensure_loaded_bounded_by_load_wait_s():
    import threading

    release = threading.Event()

    class _Manager:
        def __contains__(self, model_id):
            return False

        def load(self, model_id, api_key, **kwargs):
            release.wait(timeout=5)

    wrapper = ModelManagerGateway(_Manager(), load_wait_s=0.05)
    try:
        assert await wrapper.ensure_loaded("m") == ("load_timeout", 0)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_inner_load_timeout_error_is_load_failure_not_deadline():
    class _Manager:
        def __contains__(self, model_id):
            return False

        def load(self, model_id, api_key, **kwargs):
            raise TimeoutError("worker start timed out")

    wrapper = ModelManagerGateway(_Manager())
    assert await wrapper.load("m", "key", timeout_s=5.0) == ("error", 5)


@pytest.mark.asyncio
async def test_dead_backend_is_unloaded_and_reloaded():
    class _Manager:
        def __init__(self):
            self.healthy = False
            self.unload_calls = 0
            self.load_calls = 0

        def __contains__(self, model_id):
            return True

        def is_healthy(self, model_id):
            return self.healthy

        def unload(self, model_id):
            self.unload_calls += 1

        def load(self, model_id, api_key, **kwargs):
            self.load_calls += 1
            self.healthy = True

    mgr = _Manager()
    wrapper = ModelManagerGateway(mgr)
    assert await wrapper.ensure_loaded("m") == ("model_ready",)
    assert mgr.unload_calls == 1
    assert mgr.load_calls == 1
    assert await wrapper.ensure_loaded("m") == ("model_ready",)
    assert mgr.unload_calls == 1
    assert mgr.load_calls == 1


class _RaisingManager:
    def __init__(self, exc):
        self._exc = exc

    async def process_async(self, model_id, **kwargs):
        raise self._exc


@pytest.mark.asyncio
async def test_infer_translates_model_input_error_to_value_error():
    class ModelInputError(Exception):
        pass

    wrapper = ModelManagerGateway(_RaisingManager(ModelInputError("bad prompt shape")))
    with pytest.raises(ValueError, match="bad prompt shape"):
        await wrapper.infer(model_id="m", image=b"x")


@pytest.mark.asyncio
async def test_infer_translates_prefixed_worker_error_to_value_error():
    from inference_model_manager.errors import INPUT_ERROR_PREFIX

    wrapper = ModelManagerGateway(
        _RaisingManager(RuntimeError(INPUT_ERROR_PREFIX + "point_labels must nest"))
    )
    with pytest.raises(ValueError, match="point_labels must nest"):
        await wrapper.infer(model_id="m", image=b"x")


@pytest.mark.asyncio
async def test_infer_translates_slot_capacity_to_payload_too_large():
    from inference_server.errors import PayloadTooLargeError

    wrapper = ModelManagerGateway(
        _RaisingManager(
            ValueError("Input 999 B > slot capacity 10 B — increase input_mb")
        )
    )
    with pytest.raises(PayloadTooLargeError):
        await wrapper.infer(model_id="m", image=b"x")


@pytest.mark.asyncio
async def test_infer_translates_pool_exhaustion_to_server_busy():
    from inference_server.errors import ServerBusyError

    wrapper = ModelManagerGateway(
        _RaisingManager(TimeoutError("No free SHM slots (pool size=8)"))
    )
    with pytest.raises(ServerBusyError):
        await wrapper.infer(model_id="m", image=b"x")


@pytest.mark.asyncio
async def test_infer_deadline_raises_timeout():
    import asyncio

    class _SlowManager:
        async def process_async(self, model_id, **kwargs):
            await asyncio.sleep(5)

    wrapper = ModelManagerGateway(_SlowManager(), infer_timeout_s=0.05)
    with pytest.raises(asyncio.TimeoutError):
        await wrapper.infer(model_id="m", image=b"x")


@pytest.mark.asyncio
async def test_load_kwargs_dict_merged_into_manager_load():
    class _Manager:
        def __init__(self):
            self.load_kwargs = None

        def __contains__(self, model_id):
            return False

        def load(self, model_id, api_key, **kwargs):
            self.load_kwargs = kwargs

    mgr = _Manager()
    wrapper = ModelManagerGateway(
        mgr,
        load_kwargs={
            "backend": "subprocess",
            "decoder": "test-decoder",
            "batch_max_size": 4,
            "batch_max_delay_ms": 2.5,
        },
    )
    await wrapper.load("m", "key")
    assert mgr.load_kwargs["backend"] == "subprocess"
    assert mgr.load_kwargs["decoder"] == "test-decoder"
    assert mgr.load_kwargs["batch_max_size"] == 4
    assert mgr.load_kwargs["batch_max_delay_ms"] == 2.5


def _make_wire_manager(model_result, supports_rle=True):
    from inference_model_manager.dispatch import _get_registry
    from inference_model_manager.model_manager import ModelManager

    class _WireModel:
        def __init__(self):
            self.calls = []

        def segment(self, **kwargs):
            self.calls.append(dict(kwargs))
            return model_result

    if supports_rle:
        _WireModel.supported_mask_formats = {"rle"}

    _get_registry().register(
        _WireModel,
        "segment",
        default=True,
        validator=lambda kwargs: kwargs,
        serializer=lambda out, model: {"raw": out},
        response_type="test-v1",
    )

    class _WireBackend:
        state = "loaded"
        is_accepting = True

        def __init__(self):
            self.model = _WireModel()
            self.decoded = []

        def _decode_input(self, raw):
            self.decoded.append(raw)
            return "DECODED"

        def record_inference(self, t0, error=False):
            pass

    manager = ModelManager()
    backend = _WireBackend()
    manager._backends["wire/1"] = backend
    return manager, backend


def test_wire_marshalling_decodes_injects_rle_and_unwraps():
    manager, backend = _make_wire_manager(model_result=[["prompt-result"]])
    try:
        result = manager.process(
            "wire/1",
            task="segment",
            serialize=False,
            wire_marshalling=True,
            images=b"jpegbytes",
        )
        assert backend.decoded == [b"jpegbytes"]
        call = backend.model.calls[0]
        assert call["images"] == "DECODED"
        assert call["mask_format"] == "rle"
        assert result == ["prompt-result"]
    finally:
        manager._backends.clear()
        manager.shutdown()


def test_wire_marshalling_respects_explicit_mask_format_and_no_rle_support():
    manager, backend = _make_wire_manager(model_result=["r"], supports_rle=True)
    try:
        manager.process(
            "wire/1",
            task="segment",
            serialize=False,
            wire_marshalling=True,
            images=b"x",
            mask_format="dense",
        )
        assert backend.model.calls[0]["mask_format"] == "dense"
    finally:
        manager._backends.clear()
        manager.shutdown()

    manager2, backend2 = _make_wire_manager(model_result=["r"], supports_rle=False)
    try:
        manager2.process(
            "wire/1",
            task="segment",
            serialize=False,
            wire_marshalling=True,
            images=b"x",
        )
        assert "mask_format" not in backend2.model.calls[0]
    finally:
        manager2._backends.clear()
        manager2.shutdown()


def test_wire_marshalling_off_is_passthrough():
    manager, backend = _make_wire_manager(model_result=[["prompt-result"]])
    try:
        result = manager.process(
            "wire/1", task="segment", serialize=False, images=b"rawbytes"
        )
        assert backend.decoded == []
        call = backend.model.calls[0]
        assert call["images"] == b"rawbytes"
        assert "mask_format" not in call
        assert result == [["prompt-result"]]
    finally:
        manager._backends.clear()
        manager.shutdown()


def test_wire_marshalling_decodes_image_lists_and_maps_per_image():
    manager, backend = _make_wire_manager(model_result=["r1", "r2"])
    try:
        result = manager.process(
            "wire/1",
            task="segment",
            serialize=False,
            wire_marshalling=True,
            images=[b"a", b"b"],
        )
        assert backend.decoded == [b"a", b"b"]
        assert backend.model.calls[0]["images"] == ["DECODED", "DECODED"]
        assert result == ["r1", "r2"]
    finally:
        manager._backends.clear()
        manager.shutdown()


def test_wire_marshalling_retries_per_image_on_mismatched_batch():
    manager, backend = _make_wire_manager(model_result=["only-one"])
    try:
        result = manager.process(
            "wire/1",
            task="segment",
            serialize=False,
            wire_marshalling=True,
            images=[b"a", b"b", b"c"],
        )
        # Batched call returned 1 result for 3 images -> worker semantics
        # retry each image individually (3 extra single-image calls).
        assert len(backend.model.calls) == 4
        assert result == ["only-one", "only-one", "only-one"]
        single_calls = backend.model.calls[1:]
        assert all(call["images"] == "DECODED" for call in single_calls)
    finally:
        manager._backends.clear()
        manager.shutdown()


def test_wire_marshalling_params_only_omits_images_kwarg():
    manager, backend = _make_wire_manager(model_result=["r"])
    try:
        manager.process(
            "wire/1",
            task="segment",
            serialize=False,
            wire_marshalling=True,
            images=None,
            image_hashes=["h1"],
        )
        call = backend.model.calls[0]
        assert "images" not in call
        assert call["image_hashes"] == ["h1"]
    finally:
        manager._backends.clear()
        manager.shutdown()


def test_wire_marshalling_converts_tensors_to_numpy():
    torch = pytest.importorskip("torch")
    import numpy as np

    manager, _backend = _make_wire_manager(model_result=[{"masks": torch.ones(2, 2)}])
    try:
        result = manager.process(
            "wire/1",
            task="segment",
            serialize=False,
            wire_marshalling=True,
            images=b"x",
        )
        assert isinstance(result["masks"], np.ndarray)
    finally:
        manager._backends.clear()
        manager.shutdown()


@pytest.mark.asyncio
async def test_infer_passes_wire_marshalling_to_manager():
    mgr = _fake_manager(process_return="ok")
    wrapper = ModelManagerGateway(mgr)
    await wrapper.infer(model_id="m", image=b"x")
    assert mgr.process_async.await_args.kwargs["wire_marshalling"] is True


@pytest.mark.asyncio
async def test_real_manager_stats_shape_for_route_resolution():
    """The adapter resolves routes from stats()['models'][id]: tasks,
    model_class_name, model_mro_names, class_names, key_points_classes,
    backend_type. Pin the shape against a real ModelManager."""
    from inference_model_manager.model_manager import ModelManager
    from inference_model_manager.registry_defaults import lazy_register_by_names

    lazy_register_by_names(["SAM3Torch"])

    class _FakeBackend:
        _model_mro_names = ["SAM3Torch", "object"]
        state = "loaded"

        def stats(self):
            return {"backend_type": "subprocess", "model_class_name": "SAM3Torch"}

        @property
        def class_names(self):
            return None

        @property
        def key_points_classes(self):
            return None

    manager = ModelManager()
    try:
        manager._backends["sam3/sam3_interactive"] = _FakeBackend()
        wrapper = ModelManagerGateway(manager)
        stats = await wrapper.stats()
        entry = stats["models"]["sam3/sam3_interactive"]
        assert entry["backend_type"] == "subprocess"
        assert entry["model_class_name"] == "SAM3Torch"
        assert entry["model_mro_names"] == ["SAM3Torch", "object"]
        assert entry["class_names"] is None
        assert entry["key_points_classes"] is None
        assert "segment_with_text_prompts" in entry["tasks"]
        interface = await wrapper.interface("sam3/sam3_interactive")
        assert "embed_images" in interface["tasks"]
    finally:
        manager._backends.clear()
        manager.shutdown()


@pytest.mark.asyncio
async def test_infer_empty_image_becomes_images_none():
    mgr = _fake_manager(process_return="ok")
    wrapper = ModelManagerGateway(mgr)
    await wrapper.infer(
        model_id="sam3/sam3_final",
        image=b"",
        task="segment_with_visual_prompts",
        params={"image_hashes": ["h1"]},
    )
    kwargs = mgr.process_async.await_args.kwargs
    assert kwargs["images"] is None
    assert kwargs["image_hashes"] == ["h1"]


# ---------------------------------------------------------------------------
# multi-instance routing keys
# ---------------------------------------------------------------------------


def test_routing_key_matches_mmp_composite_format():
    assert routing_key("acme/1") == "acme/1"
    assert routing_key("acme/1", "") == "acme/1"
    assert routing_key("acme/1", "b") == "acme/1:b"


def test_routed_model_id_strips_instance_suffix():
    assert routed_model_id("acme/1") == "acme/1"
    assert routed_model_id("acme/1:b") == "acme/1"


@pytest.mark.asyncio
async def test_infer_routes_by_instance():
    mgr = _fake_manager(process_return="ok")
    wrapper = ModelManagerGateway(mgr)
    await wrapper.infer(model_id="acme/1", image=b"x", instance="b")
    assert mgr.process_async.await_args.args[0] == "acme/1:b"


@pytest.mark.asyncio
async def test_infer_without_instance_keeps_bare_model_id():
    mgr = _fake_manager(process_return="ok")
    wrapper = ModelManagerGateway(mgr)
    await wrapper.infer(model_id="acme/1", image=b"x")
    assert mgr.process_async.await_args.args[0] == "acme/1"


@pytest.mark.asyncio
async def test_ensure_loaded_registers_instance_under_composite_key():
    mgr = _fake_manager()
    mgr.__contains__ = MagicMock(return_value=False)
    wrapper = ModelManagerGateway(mgr)
    status = await wrapper.ensure_loaded("acme/1", "b", "key")
    assert status[0] == "model_ready"
    args, kwargs = mgr.load.call_args
    assert args[0] == "acme/1:b"
    assert kwargs["model_id_or_path"] == "acme/1"


@pytest.mark.asyncio
async def test_ensure_loaded_without_instance_passes_no_path_override():
    mgr = _fake_manager()
    mgr.__contains__ = MagicMock(return_value=False)
    wrapper = ModelManagerGateway(mgr)
    await wrapper.ensure_loaded("acme/1", "", "key")
    args, kwargs = mgr.load.call_args
    assert args[0] == "acme/1"
    assert "model_id_or_path" not in kwargs


@pytest.mark.asyncio
async def test_ensure_loaded_isolates_instances_from_each_other():
    mgr = _fake_manager()
    present: set[str] = set()
    mgr.__contains__ = MagicMock(side_effect=lambda key: key in present)
    mgr.load = MagicMock(side_effect=lambda key, *a, **kw: present.add(key))
    mgr.is_healthy = MagicMock(return_value=True)
    wrapper = ModelManagerGateway(mgr)

    await wrapper.ensure_loaded("acme/1", "b", "key")
    await wrapper.ensure_loaded("acme/1", "c", "key")
    await wrapper.ensure_loaded("acme/1", "b", "key")

    assert present == {"acme/1:b", "acme/1:c"}
    assert [call.args[0] for call in mgr.load.call_args_list] == [
        "acme/1:b",
        "acme/1:c",
    ]


@pytest.mark.asyncio
async def test_load_admin_route_splits_instance_suffix_for_weights():
    mgr = _fake_manager()
    mgr.__contains__ = MagicMock(return_value=False)
    wrapper = ModelManagerGateway(mgr)
    assert (await wrapper.load("acme/1:b", "key"))[0] == "ok"
    args, kwargs = mgr.load.call_args
    assert args[0] == "acme/1:b"
    assert kwargs["model_id_or_path"] == "acme/1"
