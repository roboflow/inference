"""Security regressions at SQL, image and request allocation boundaries."""

import io
import json
import time
from unittest.mock import Mock

import pytest
from PIL import Image
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import v1
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source.query import (
    build_image_query,
)
from roboflow_workflows.utils import subprocess_supervisor as supervisor


def query(**overrides):
    values = dict(
        catalog="main",
        schema="vision",
        table="images",
        image_column="bytes",
        image_name_column="id",
        metadata_columns=["note"],
        max_rows=100,
    )
    values.update(overrides)
    return build_image_query(**values)


@pytest.mark.parametrize(
    "field", ["catalog", "schema", "table", "image_column", "image_name_column"]
)
def test_sql_fragments_remain_one_quoted_identifier_parameter(field):
    attack = "a`; DROP TABLE main.vision.images; --"
    baseline, _ = query()
    statement, parameters = query(**{field: attack})
    assert statement == baseline
    assert attack not in statement
    assert any(
        "`a``; DROP TABLE main.vision.images; --`" in str(value)
        for value in parameters.values()
    )


@pytest.mark.parametrize("value", [0, 1001, True, "100", "1; DROP TABLE t"])
def test_limit_rejects_values_before_query_execution(value):
    with pytest.raises(ValueError):
        query(max_rows=value)


def test_names_are_components_not_qualified_expressions():
    _, parameters = query(table="other.schema.table", image_column="*")
    assert parameters["source"] == "`main`.`vision`.`other.schema.table`"
    assert parameters["column_0"] == "`*`"


def test_removed_sql_input_is_rejected_by_manifest():
    with pytest.raises(ValueError, match="sql_statement"):
        v1.BlockManifest.model_validate(
            dict(
                type="roboflow_core/databricks_sql_image_source@v1",
                name="db",
                server_hostname="dbc-test.cloud.databricks.com",
                http_path="/sql/1.0/warehouses/abc",
                access_token="token",
                catalog="main",
                schema_name="vision",
                table="images",
                sql_statement="DROP TABLE t",
            )
        )


def test_webp_rejected_before_constructing_native_decoder(monkeypatch):
    stream = io.BytesIO()
    Image.new("RGB", (2048, 2048)).save(stream, format="WEBP", lossless=True)
    from PIL import _webp

    decoder = Mock(side_effect=AssertionError("native allocation"))
    monkeypatch.setattr(_webp, "WebPAnimDecoder", decoder)
    with pytest.raises(ValueError, match="not allowed"):
        v1.decode_image(stream.getvalue(), budget=v1.RunBudget(0))
    decoder.assert_not_called()


def test_oversized_request_rejected_before_json_serialization_or_process(monkeypatch):
    monkeypatch.setattr(
        supervisor.json, "dumps", Mock(side_effect=AssertionError("serialized"))
    )
    monkeypatch.setattr(
        supervisor.subprocess, "Popen", Mock(side_effect=AssertionError("launched"))
    )
    with pytest.raises(ValueError, match="byte limit"):
        supervisor.run_in_subprocess(
            {"token": "\U0001f600" * (1024 * 1024)},
            worker_module="never",
            deadline=time.monotonic() + 5,
            read_result=lambda _: None,
        )


@pytest.mark.parametrize(
    "value",
    [
        {"x": '\U0001f600\n\\"\x00é'},
        [True, False, None, -123, 1.25],
        {"a": [], "b": {}},
    ],
)
def test_bounded_json_counts_escaped_utf8_bytes(value):
    expected = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert supervisor.bounded_json_bytes(value, max_bytes=len(expected)) == expected
    with pytest.raises(ValueError):
        supervisor.bounded_json_bytes(value, max_bytes=len(expected) - 1)


def test_thrift_declared_lengths_fail_before_reading_payload():
    import struct

    from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
        transport,
    )
    from thrift.protocol import TBinaryProtocol
    from thrift.transport.TTransport import TMemoryBuffer, TTransportException

    transport.install_pinned_transport()
    buffer = TMemoryBuffer(struct.pack("!i", 2**30))
    buffer._roboflow_destination = transport.PinnedDestination(
        "test.cloud.databricks.com", "8.8.8.8", max_response_bytes=1024
    )
    protocol = TBinaryProtocol.TBinaryProtocol(buffer)
    with pytest.raises(TTransportException, match="Length exceeded"):
        protocol.readBinary()
    buffer = TMemoryBuffer(b"\x0b" + struct.pack("!i", 1000000))
    buffer._roboflow_destination = transport.PinnedDestination(
        "test.cloud.databricks.com", "8.8.8.8"
    )
    with pytest.raises(TTransportException, match="Length exceeded"):
        TBinaryProtocol.TBinaryProtocol(buffer).readListBegin()


def test_arrow_schema_is_refused_before_native_parsing():
    from databricks.sql.thrift_api.TCLIService import ttypes
    from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
        transport,
    )
    from thrift.protocol import TBinaryProtocol
    from thrift.transport.TTransport import TMemoryBuffer

    transport.install_pinned_transport()
    output = TMemoryBuffer()
    ttypes.TGetResultSetMetadataResp(arrowSchema=b"not an arrow schema").write(
        TBinaryProtocol.TBinaryProtocol(output)
    )
    buffer = TMemoryBuffer(output.getvalue())
    buffer._roboflow_destination = transport.PinnedDestination(
        "test.cloud.databricks.com", "8.8.8.8"
    )
    with pytest.raises(transport.PinnedTransportError, match="column-based"):
        ttypes.TGetResultSetMetadataResp().read(TBinaryProtocol.TBinaryProtocol(buffer))
