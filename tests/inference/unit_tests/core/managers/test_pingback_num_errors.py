from unittest import mock

from inference.core.managers import base as base_module
from inference.core.managers import pingback as pingback_module
from inference.core.managers.base import ModelManager, ModelRegistry
from inference.core.managers.decorators.base import ModelManagerDecorator


def test_increment_num_errors():
    mm = ModelManager(ModelRegistry(dict()))
    mm_wrapper = ModelManagerDecorator(mm)
    # Do not start the real pingback scheduler: it would keep posting metrics
    # to Roboflow from a background thread for the rest of the pytest session.
    with mock.patch.object(base_module, "METRICS_ENABLED", False):
        mm_wrapper.init_pingback()
    assert mm.pingback is None
    mm_wrapper.num_errors += 1
    assert mm.num_errors == mm_wrapper.num_errors == 1
    mm.num_errors += 1
    assert mm.num_errors == mm_wrapper.num_errors == 2


def test_pingback_post_is_noop_in_offline_mode():
    pingback = pingback_module.PingbackInfo.__new__(pingback_module.PingbackInfo)

    with mock.patch.object(pingback_module, "OFFLINE_MODE", True), mock.patch.object(
        pingback_module.requests, "post"
    ) as post:
        pingback.post_data(model_manager=mock.MagicMock())

    post.assert_not_called()


def test_pingback_start_registers_atexit_shutdown_and_stop_is_idempotent():
    pingback = pingback_module.PingbackInfo.__new__(pingback_module.PingbackInfo)
    pingback.scheduler = mock.MagicMock()
    pingback.scheduler.running = True
    pingback.model_manager = mock.MagicMock()

    with mock.patch.object(pingback_module, "METRICS_ENABLED", True), mock.patch.object(
        pingback_module.atexit, "register"
    ) as register:
        pingback.start()

    register.assert_called_once_with(pingback.stop, wait=False)

    pingback.stop(wait=False)
    pingback.scheduler.shutdown.assert_called_once_with(wait=False)

    # a stopped (or never started) scheduler must be a no-op on later calls
    pingback.scheduler.running = False
    pingback.stop()
    pingback.scheduler.shutdown.assert_called_once()
