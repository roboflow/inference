from unittest.mock import MagicMock, patch

import pytest
import requests

from inference.enterprise.workflows.enterprise_blocks.sinks.plc_relay.v1 import (
    PLCRelayReaderBlockV1,
    PLCRelayWriterBlockV1,
)

CLIENT_MODULE = (
    "inference.enterprise.workflows.enterprise_blocks.sinks.plc_relay.client.requests"
)
BLOCK_MODULE = (
    "inference.enterprise.workflows.enterprise_blocks.sinks.plc_relay.v1.requests"
)


def _make_response(status_code=200, json_body=None, text=""):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_body if json_body is not None else {}
    resp.text = text
    return resp


def _malformed_response(status_code=200, text="not json"):
    """A 200 whose body cannot be parsed as a JSON object."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.side_effect = ValueError("Expecting value")
    resp.text = text
    return resp


def _read_batch_body(values):
    """Build a /read_batch 200 body from a {tag: value} mapping (value may be a dict
    with an 'error' key to simulate a per-tag read error)."""
    tags = []
    for name, value in values.items():
        if isinstance(value, dict):
            tags.append({"name": name, **value})
        else:
            tags.append({"name": name, "value": value})
    return {"tags": tags, "count": len(tags)}


def _write_batch_body(results):
    """Build a /write_batch 200 body from a {tag: success_bool_or_dict} mapping."""
    out = []
    for name, result in results.items():
        if isinstance(result, dict):
            out.append({"name": name, **result})
        else:
            out.append({"name": name, "success": result})
    return {
        "results": out,
        "success_count": sum(1 for r in out if r.get("success")),
        "error_count": sum(1 for r in out if not r.get("success")),
    }


def _session(block_requests, client_requests):
    """Wire one session mock onto both patched ``requests`` modules.

    The block creates its session via ``requests.Session()`` (v1 module) and reuses it
    across runs; the client helpers call ``session.post`` (client module). The client
    module's ``requests.exceptions`` must stay real so exception handling works.
    """
    client_requests.exceptions = requests.exceptions
    session = MagicMock()
    block_requests.Session.return_value = session
    return session


# ----------------------------- Reader -----------------------------


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_reads_tags(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)
    # The relay returns only bool | int | float | null tag values.
    session.post.return_value = _make_response(
        json_body=_read_batch_body({"camera_ready": True, "sku_number": 42})
    )

    result = PLCRelayReaderBlockV1().run(tags_to_read=["camera_ready", "sku_number"])

    assert result["tag_values"] == {"camera_ready": True, "sku_number": 42}
    assert result["error_status"] is False
    # One batch call to the on-device relay's /read_batch endpoint.
    assert session.post.call_count == 1
    call = session.post.call_args
    assert call.args[0] == "http://localhost:8007/read_batch"
    assert call.kwargs["json"] == {"tags": ["camera_ready", "sku_number"]}


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_no_tags_makes_no_call(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)

    result = PLCRelayReaderBlockV1().run(tags_to_read=[])

    assert result == {"tag_values": {}, "error_status": False}
    session.post.assert_not_called()


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_batch_rejected_marks_all_failed(block_requests, client_requests) -> None:
    # An unknown tag yields a 404 for the whole batch; every tag is a failure.
    session = _session(block_requests, client_requests)
    session.post.return_value = _make_response(
        status_code=404, json_body={"detail": "Unknown tag: bad_tag"}
    )

    result = PLCRelayReaderBlockV1().run(tags_to_read=["good_tag", "bad_tag"])

    assert result["tag_values"] == {
        "good_tag": "ReadFailure",
        "bad_tag": "ReadFailure",
    }
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_per_tag_error_in_batch(block_requests, client_requests) -> None:
    # A successful (200) batch can still carry a per-tag error.
    session = _session(block_requests, client_requests)
    session.post.return_value = _make_response(
        json_body=_read_batch_body(
            {"good_tag": 100, "bad_tag": {"value": None, "error": "Not connected"}}
        )
    )

    result = PLCRelayReaderBlockV1().run(tags_to_read=["good_tag", "bad_tag"])

    assert result["tag_values"]["good_tag"] == 100
    assert result["tag_values"]["bad_tag"] == "ReadFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_missing_tag_in_response(block_requests, client_requests) -> None:
    # The relay omits a requested tag from the result array.
    session = _session(block_requests, client_requests)
    session.post.return_value = _make_response(
        json_body=_read_batch_body({"present_tag": 1})
    )

    result = PLCRelayReaderBlockV1().run(tags_to_read=["present_tag", "missing_tag"])

    assert result["tag_values"]["present_tag"] == 1
    assert result["tag_values"]["missing_tag"] == "ReadFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_connection_error(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)
    session.post.side_effect = requests.exceptions.ConnectionError("refused")

    result = PLCRelayReaderBlockV1().run(tags_to_read=["unreachable_tag"])

    assert result["tag_values"]["unreachable_tag"] == "ReadFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_malformed_success_body(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)
    session.post.return_value = _malformed_response()

    result = PLCRelayReaderBlockV1().run(tags_to_read=["weird_tag"])

    assert result["tag_values"]["weird_tag"] == "ReadFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_real_value_matching_sentinel(block_requests, client_requests) -> None:
    # A real tag value that happens to equal the failure sentinel must not be
    # reported as an error.
    session = _session(block_requests, client_requests)
    session.post.return_value = _make_response(
        json_body=_read_batch_body({"status_tag": "ReadFailure"})
    )

    result = PLCRelayReaderBlockV1().run(tags_to_read=["status_tag"])

    assert result["tag_values"]["status_tag"] == "ReadFailure"
    assert result["error_status"] is False


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_custom_url_strips_trailing_slash(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)
    session.post.return_value = _make_response(
        json_body=_read_batch_body({"good_read": 100})
    )

    result = PLCRelayReaderBlockV1().run(
        tags_to_read=["good_read"],
        relay_url="http://192.168.1.10:8007/",
        request_timeout=10,
    )

    assert result["tag_values"]["good_read"] == 100
    assert session.post.call_args.args[0] == "http://192.168.1.10:8007/read_batch"


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_reader_session_reused_across_frames(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)
    session.post.return_value = _make_response(
        json_body=_read_batch_body({"t": 1})
    )

    block = PLCRelayReaderBlockV1()
    block.run(tags_to_read=["t"])
    block.run(tags_to_read=["t"])

    # The session is created once and reused across runs (HTTP keep-alive).
    assert block_requests.Session.call_count == 1
    assert session.post.call_count == 2


# ----------------------------- Writer -----------------------------


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_writer_writes_tags(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)
    session.post.return_value = _make_response(
        json_body=_write_batch_body({"camera_fault": True, "defect_count": True})
    )

    result = PLCRelayWriterBlockV1().run(
        tags_to_write={"camera_fault": True, "defect_count": 5},
        depends_on=None,
    )

    assert result["write_results"]["camera_fault"] == "WriteSuccess"
    assert result["write_results"]["defect_count"] == "WriteSuccess"
    assert result["error_status"] is False
    assert session.post.call_count == 1
    call = session.post.call_args
    assert call.args[0] == "http://localhost:8007/write_batch"
    assert call.kwargs["json"] == {
        "writes": [
            {"name": "camera_fault", "value": True},
            {"name": "defect_count", "value": 5},
        ]
    }


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_writer_batch_rejected_marks_all_failed(block_requests, client_requests) -> None:
    # A non-writable tag yields a 403 for the whole batch; every tag is a failure.
    session = _session(block_requests, client_requests)
    session.post.return_value = _make_response(
        status_code=403, json_body={"detail": "Tag 'ro_tag' is not writable"}
    )

    result = PLCRelayWriterBlockV1().run(
        tags_to_write={"ok_tag": 1, "ro_tag": 99}, depends_on=None
    )

    assert result["write_results"] == {
        "ok_tag": "WriteFailure",
        "ro_tag": "WriteFailure",
    }
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_writer_per_tag_value_rejected(block_requests, client_requests) -> None:
    # A successful (200) batch can still carry a per-tag value-rejection error.
    session = _session(block_requests, client_requests)
    session.post.return_value = _make_response(
        json_body=_write_batch_body(
            {
                "ok_tag": True,
                "int_tag": {
                    "success": False,
                    "error": "INT value 99999 out of range (-32768 to 32767)",
                },
            }
        )
    )

    result = PLCRelayWriterBlockV1().run(
        tags_to_write={"ok_tag": 1, "int_tag": 99999}, depends_on=None
    )

    assert result["write_results"]["ok_tag"] == "WriteSuccess"
    assert result["write_results"]["int_tag"] == "WriteFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_writer_malformed_success_body(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)
    session.post.return_value = _malformed_response()

    result = PLCRelayWriterBlockV1().run(
        tags_to_write={"weird_tag": 1}, depends_on=None
    )

    assert result["write_results"]["weird_tag"] == "WriteFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_writer_disabled(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)

    result = PLCRelayWriterBlockV1().run(
        tags_to_write={"camera_fault": True},
        depends_on=None,
        disable_sink=True,
    )

    assert result["write_results"] == {}
    assert result["error_status"] is False
    session.post.assert_not_called()


@pytest.mark.timeout(10)
@patch(CLIENT_MODULE)
@patch(BLOCK_MODULE)
def test_writer_no_tags_makes_no_call(block_requests, client_requests) -> None:
    session = _session(block_requests, client_requests)

    result = PLCRelayWriterBlockV1().run(tags_to_write={}, depends_on=None)

    assert result == {"write_results": {}, "error_status": False}
    session.post.assert_not_called()
