from unittest.mock import patch

import pytest

from inference_sdk.config import INTERNAL_REMOTE_EXEC_REQ_HEADER, execution_id
from inference_sdk.http.utils.request_building import (
    ImagePlacement,
    RequestData,
    assembly_request_data,
    prepare_requests_data,
)


def test_assembly_request_data_when_image_placement_is_in_body_and_single_image_given() -> (
    None
):
    # when
    result = assembly_request_data(
        url="https://some.com",
        batch_inference_inputs=[("image_1", None)],
        headers={"some": "header"},
        parameters={"api_key": "secret"},
        payload=None,
        image_placement=ImagePlacement.DATA,
    )

    # then
    assert result == RequestData(
        url="https://some.com",
        request_elements=1,
        headers={"some": "header"},
        parameters={"api_key": "secret"},
        data="image_1",
        payload=None,
        image_scaling_factors=[None],
    )


def test_assembly_request_data_when_image_placement_is_in_body_and_batch_of_images_given() -> (
    None
):
    # when
    with pytest.raises(ValueError):
        _ = assembly_request_data(
            url="https://some.com",
            batch_inference_inputs=[("image_1", None), ("image_2", None)],
            headers={"some": "header"},
            parameters={"api_key": "secret"},
            payload=None,
            image_placement=ImagePlacement.DATA,
        )


def test_assembly_request_data_when_image_placement_is_in_json_and_single_image_given() -> (
    None
):
    # when
    result = assembly_request_data(
        url="https://some.com",
        batch_inference_inputs=[("image_1", 1.0)],
        headers={"some": "header"},
        parameters=None,
        payload={"api_key": "secret"},
        image_placement=ImagePlacement.JSON,
    )

    # then
    assert result == RequestData(
        url="https://some.com",
        request_elements=1,
        headers={"some": "header"},
        parameters=None,
        data=None,
        payload={"api_key": "secret", "image": {"type": "base64", "value": "image_1"}},
        image_scaling_factors=[1.0],
    )


def test_assembly_request_data_when_image_placement_is_in_json_and_multiple_images_given() -> (
    None
):
    # when
    result = assembly_request_data(
        url="https://some.com",
        batch_inference_inputs=[("image_1", 1.0), ("image_2", 0.5)],
        headers={"some": "header"},
        parameters=None,
        payload={"api_key": "secret"},
        image_placement=ImagePlacement.JSON,
    )

    # then
    assert result == RequestData(
        url="https://some.com",
        request_elements=2,
        headers={"some": "header"},
        parameters=None,
        data=None,
        payload={
            "api_key": "secret",
            "image": [
                {"type": "base64", "value": "image_1"},
                {"type": "base64", "value": "image_2"},
            ],
        },
        image_scaling_factors=[1.0, 0.5],
    )


def test_prepare_requests_data() -> None:
    # when
    result = prepare_requests_data(
        url="https://some.com",
        encoded_inference_inputs=[
            ("image_1", 1.0),
            ("image_2", 0.5),
            ("image_3", 0.75),
        ],
        headers={"some": "header"},
        parameters=None,
        payload={"api_key": "secret"},
        max_batch_size=2,
        image_placement=ImagePlacement.JSON,
    )

    # then
    assert (
        len(result) == 2
    ), "Input is expected to be split into two requests, as max size of batch equals to 2"
    assert result[0] == RequestData(
        url="https://some.com",
        request_elements=2,
        headers={"some": "header"},
        parameters=None,
        data=None,
        payload={
            "api_key": "secret",
            "image": [
                {"type": "base64", "value": "image_1"},
                {"type": "base64", "value": "image_2"},
            ],
        },
        image_scaling_factors=[1.0, 0.5],
    )
    assert result[1] == RequestData(
        url="https://some.com",
        request_elements=1,
        headers={"some": "header"},
        parameters=None,
        data=None,
        payload={
            "api_key": "secret",
            "image": {"type": "base64", "value": "image_3"},
        },
        image_scaling_factors=[0.75],
    )


class TestInternalRemoteExecHeader:
    @patch(
        "inference_sdk.http.utils.request_building.ENABLE_INTERNAL_REMOTE_EXEC_HEADER",
        True,
    )
    @patch.dict("os.environ", {"ROBOFLOW_INTERNAL_SERVICE_SECRET": "my-secret"})
    def test_header_sent_when_enabled_and_execution_id_set(self) -> None:
        """When toggle is on, secret is set, and execution_id is active,
        the internal execution header should be injected."""
        token = execution_id.set("test-exec-id")
        try:
            result = assembly_request_data(
                url="https://some.com",
                batch_inference_inputs=[("image_1", None)],
                headers={"some": "header"},
                parameters=None,
                payload=None,
                image_placement=ImagePlacement.DATA,
            )
            assert result.headers[INTERNAL_REMOTE_EXEC_REQ_HEADER] == "my-secret"
        finally:
            execution_id.reset(token)

    @patch(
        "inference_sdk.http.utils.request_building.ENABLE_INTERNAL_REMOTE_EXEC_HEADER",
        False,
    )
    @patch.dict("os.environ", {"ROBOFLOW_INTERNAL_SERVICE_SECRET": "my-secret"})
    def test_header_not_sent_when_toggle_disabled(self) -> None:
        """When toggle is off, the header should not be sent even if
        secret and execution_id are present."""
        token = execution_id.set("test-exec-id")
        try:
            result = assembly_request_data(
                url="https://some.com",
                batch_inference_inputs=[("image_1", None)],
                headers={"some": "header"},
                parameters=None,
                payload=None,
                image_placement=ImagePlacement.DATA,
            )
            assert INTERNAL_REMOTE_EXEC_REQ_HEADER not in (result.headers or {})
        finally:
            execution_id.reset(token)

    @patch(
        "inference_sdk.http.utils.request_building.ENABLE_INTERNAL_REMOTE_EXEC_HEADER",
        True,
    )
    @patch.dict("os.environ", {}, clear=False)
    def test_header_not_sent_when_no_secret(self) -> None:
        """When toggle is on but no secret is configured, the header should not
        be sent."""
        import os

        os.environ.pop("ROBOFLOW_INTERNAL_SERVICE_SECRET", None)
        token = execution_id.set("test-exec-id")
        try:
            result = assembly_request_data(
                url="https://some.com",
                batch_inference_inputs=[("image_1", None)],
                headers={"some": "header"},
                parameters=None,
                payload=None,
                image_placement=ImagePlacement.DATA,
            )
            assert INTERNAL_REMOTE_EXEC_REQ_HEADER not in (result.headers or {})
        finally:
            execution_id.reset(token)

    @patch(
        "inference_sdk.http.utils.request_building.ENABLE_INTERNAL_REMOTE_EXEC_HEADER",
        True,
    )
    @patch.dict("os.environ", {"ROBOFLOW_INTERNAL_SERVICE_SECRET": "my-secret"})
    def test_header_not_sent_when_no_execution_id(self) -> None:
        """When toggle is on and secret is set but no execution_id is active,
        the header should not be sent (not in a server request context)."""
        token = execution_id.set(None)
        try:
            result = assembly_request_data(
                url="https://some.com",
                batch_inference_inputs=[("image_1", None)],
                headers={"some": "header"},
                parameters=None,
                payload=None,
                image_placement=ImagePlacement.DATA,
            )
            assert INTERNAL_REMOTE_EXEC_REQ_HEADER not in (result.headers or {})
        finally:
            execution_id.reset(token)
