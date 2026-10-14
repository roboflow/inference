from unittest import mock

import numpy as np
import pytest
from roboflow_workflows.core_steps.sinks.roboflow.vision_events import (
    v1 as vision_events_v1,
)
from roboflow_workflows.core_steps.sinks.roboflow.vision_events import (
    v1_tensor as vision_events_v1_tensor,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.prototypes.platform_client import OFFLINE_PLATFORM_CLIENT

VISION_EVENTS_MODULES = [vision_events_v1, vision_events_v1_tensor]


class _HeadersOnlyClient:
    def build_api_headers(self, explicit_headers=None):
        return dict(explicit_headers or {})

    def wrap_url(self, url: str) -> str:
        return url


def _image() -> WorkflowImageData:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((8, 12, 3), dtype=np.uint8),
    )
    return image


def _run_block(module, platform_client, *, with_image: bool, **run_parameters):
    block = module.RoboflowVisionEventsBlockV1(
        api_key="my-key",
        background_tasks=None,
        thread_pool_executor=None,
        platform_client=platform_client,
    )
    result = block.run(
        input_image=_image() if with_image else None,
        output_image=_image() if with_image else None,
        predictions=None,
        event_type="custom",
        solution="my-solution",
        custom_metadata={},
        fire_and_forget=False,
        disable_sink=False,
        cooldown_seconds=0,
        **run_parameters,
    )
    return result


@pytest.mark.parametrize("with_image", [True, False])
@pytest.mark.parametrize(
    "platform_client", [OFFLINE_PLATFORM_CLIENT, _HeadersOnlyClient()]
)
@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_cloud_vision_event_is_refused_without_roboflow_api_access(
    module, platform_client, with_image: bool
) -> None:
    # when
    with mock.patch.object(module.requests, "post") as post_mock:
        result = _run_block(module, platform_client, with_image=with_image)

    # then
    post_mock.assert_not_called()
    assert result["error_status"] is True
    assert result["event_id"] == ""
    assert "calls the Roboflow API, which is not available" in result["message"]


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_upload_image_is_refused_without_roboflow_api_access(module) -> None:
    # when
    with mock.patch.object(module.requests, "post") as post_mock:
        with pytest.raises(Exception, match="calls the Roboflow API"):
            module._upload_image(
                "https://api.roboflow.com",
                "my-key",
                _image(),
                platform_client=OFFLINE_PLATFORM_CLIENT,
            )

    # then
    post_mock.assert_not_called()


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_event_store_path_still_works_with_the_offline_client(module) -> None:
    # given
    response = mock.MagicMock(status_code=201)
    response.json.return_value = {"id": "event-1"}

    # when
    with mock.patch.object(module.requests, "post", return_value=response) as post_mock:
        result = _run_block(
            module,
            OFFLINE_PLATFORM_CLIENT,
            with_image=True,
            write_to_event_store=True,
            event_store_url="http://edge:8001",
        )

    # then
    assert result["error_status"] is False
    assert result["event_id"] == "event-1"
    post_mock.assert_called_once()
    assert post_mock.call_args.args[0] == "http://edge:8001/v2/events"
