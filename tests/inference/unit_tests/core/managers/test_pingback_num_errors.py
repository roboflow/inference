from unittest import mock

from inference.core.managers import pingback as pingback_module
from inference.core.managers.base import ModelManager, ModelRegistry
from inference.core.managers.decorators.base import ModelManagerDecorator


def test_increment_num_errors():
    mm = ModelManager(ModelRegistry(dict()))
    mm_wrapper = ModelManagerDecorator(mm)
    mm_wrapper.init_pingback()
    mm_wrapper.num_errors += 1
    assert mm.num_errors == mm_wrapper.num_errors == 1
    mm.num_errors += 1
    assert mm.num_errors == mm_wrapper.num_errors == 2


def test_pingback_post_is_noop_in_offline_mode():
    pingback = pingback_module.PingbackInfo.__new__(pingback_module.PingbackInfo)

    with mock.patch.object(
        pingback_module, "OFFLINE_MODE", True
    ), mock.patch.object(pingback_module.requests, "post") as post:
        pingback.post_data(model_manager=mock.MagicMock())

    post.assert_not_called()
