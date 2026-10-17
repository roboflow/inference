import os
import subprocess
import sys
import textwrap
from unittest import mock

from inference.core.managers import pingback as pingback_module
from inference.core.managers.base import ModelManager, ModelRegistry
from inference.core.managers.decorators.base import ModelManagerDecorator


@mock.patch("inference.core.managers.base.PingbackInfo")
def test_increment_num_errors(_pingback):
    mm = ModelManager(ModelRegistry(dict()))
    mm_wrapper = ModelManagerDecorator(mm)
    mm_wrapper.init_pingback()
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


def _pingback_for_test():
    with mock.patch.object(pingback_module, "BackgroundScheduler"), mock.patch.object(
        pingback_module, "get_system_info", return_value={}
    ):
        return pingback_module.PingbackInfo(manager=mock.MagicMock(uuid="uuid"))


def _posted_api_key(pingback, fallback_api_key=None):
    pingback.fallback_api_key = fallback_api_key
    model_manager = mock.MagicMock()
    model_manager.models.return_value = []
    with mock.patch.object(pingback_module, "OFFLINE_MODE", False), mock.patch.object(
        pingback_module.requests, "post"
    ) as post:
        pingback.post_data(model_manager=model_manager)
    return post.call_args.kwargs["json"]["api_key"]


def test_pingback_sends_metrics_api_key_over_borrowed_inbound_key():
    with mock.patch.object(
        pingback_module, "METRICS_API_KEY", "metrics-key"
    ), mock.patch.object(pingback_module, "API_KEY", None):
        pingback = _pingback_for_test()

    assert _posted_api_key(pingback, fallback_api_key="inbound-key") == "metrics-key"


def test_pingback_without_metrics_api_key_keeps_previous_behaviour():
    with mock.patch.object(pingback_module, "METRICS_API_KEY", None), mock.patch.object(
        pingback_module, "API_KEY", None
    ):
        pingback = _pingback_for_test()

    assert _posted_api_key(pingback, fallback_api_key="inbound-key") == "inbound-key"


def test_pingback_does_not_log_the_api_key_when_posting_fails():
    with mock.patch.object(pingback_module, "METRICS_API_KEY", "metrics-key"):
        pingback = _pingback_for_test()
    model_manager = mock.MagicMock()
    model_manager.models.return_value = []

    with mock.patch.object(pingback_module, "OFFLINE_MODE", False), mock.patch.object(
        pingback_module.requests, "post", side_effect=ConnectionError("down")
    ), mock.patch.object(pingback_module.logger, "exception") as log_exception:
        pingback.post_data(model_manager=model_manager)

    logged = " ".join(str(arg) for arg in log_exception.call_args.args)
    assert "down" in logged
    assert "metrics-key" not in logged


def test_metrics_api_key_is_not_a_fallback_credential_for_requests():
    # Fresh interpreter: env.py binds API_KEY at import time.
    script = textwrap.dedent("""
        from inference.core import env
        from inference.core.interfaces.http.api_key_resolution import (
            api_key_fallback,
        )
        from inference.core.managers.decorators import base as decorators_base
        from inference.core.models import roboflow as roboflow_models

        assert env.METRICS_API_KEY == "metrics-key", env.METRICS_API_KEY
        assert env.API_KEY is None
        assert env.PRELOAD_API_KEY is None
        assert decorators_base.API_KEY is None
        assert roboflow_models.API_KEY is None
        assert api_key_fallback(None) is None
        """)
    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in {"ROBOFLOW_API_KEY", "API_KEY", "PRELOAD_API_KEY"}
    }
    environment["METRICS_API_KEY"] = "metrics-key"

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr[-2000:]
