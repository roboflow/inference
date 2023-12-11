import datetime
import json
from enum import Enum

from inference.enterprise.stream_management.manager.entities import (
    ErrorType,
    OperationStatus,
)
from inference.enterprise.stream_management.manager.serialisation import (
    describe_error,
    prepare_error_response,
    serialise_to_json,
)


def test_serialise_to_json_when_datetime_object_given() -> None:
    # given
    timestamp = datetime.datetime(
        year=2020, month=10, day=13, hour=10, minute=30, second=12
    )

    # when
    serialised = json.dumps({"time": timestamp}, default=serialise_to_json)

    # then

    assert (
        "2020-10-13T10:30:12" in serialised
    ), "Timestamp in format YYYY-MM-DDTHH:MM:HH must be present in serialised json"


def test_serialise_to_json_when_date_object_given() -> None:
    # given
    timestamp = datetime.date(year=2020, month=10, day=13)

    # when
    serialised = json.dumps({"time": timestamp}, default=serialise_to_json)

    # then

    assert (
        "2020-10-13" in serialised
    ), "Date in format YYYY-MM-DD must be present in serialised json"


class ExampleEnum(Enum):
    SOME = "some"
    OTHER = "other"


def test_serialise_to_json_when_enum_object_given() -> None:
    # given
    data = ExampleEnum.SOME

    # when
    serialised = json.dumps({"payload": data}, default=serialise_to_json)

    # then

    assert "some" in serialised, "Enum value `some` must be present in serialised json"


def test_serialise_to_json_when_no_special_content_given() -> None:
    # given
    data = {"some": 1, "other": True}

    # when
    serialised = json.dumps(data, default=serialise_to_json)
    result = json.loads(serialised)

    # then

    assert result == data, "After deserialization, data must be recovered 100%"


def test_describe_error_when_exception_is_provided_as_context() -> None:
    # given
    exception = ValueError("Some value error")

    # when
    result = describe_error(exception=exception, error_type=ErrorType.INVALID_PAYLOAD)

    # then
    assert result == {
        "status": OperationStatus.FAILURE,
        "error_type": ErrorType.INVALID_PAYLOAD,
        "error_class": "ValueError",
        "error_message": "Some value error",
    }


def test_describe_error_when_exception_is_not_provided() -> None:
    # when
    result = describe_error(exception=None, error_type=ErrorType.INVALID_PAYLOAD)

    # then
    assert result == {
        "status": OperationStatus.FAILURE,
        "error_type": ErrorType.INVALID_PAYLOAD,
    }


def test_prepare_error_response() -> None:
    # given
    exception = ValueError("Some value error")

    # when
    error_response = prepare_error_response(
        request_id="my_request",
        error=exception,
        error_type=ErrorType.INTERNAL_ERROR,
        pipeline_id="my_pipeline",
    )
    decoded_response = json.loads(error_response.decode("utf-8"))

    # then
    assert decoded_response == {
        "request_id": "my_request",
        "response": {
            "status": "failure",
            "error_type": "internal_error",
            "error_class": "ValueError",
            "error_message": "Some value error",
        },
        "pipeline_id": "my_pipeline",
    }
