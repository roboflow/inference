"""Subprocess lifecycle and bounded output tests; no warehouse required."""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
    isolation,
    v1,
)

LIMITS = dict(
    max_run_bytes=100000,
    max_image_bytes=32000,
    max_image_pixels=1000,
    allow_non_global=False,
    hosted=False,
)
REQUEST = dict(
    server_hostname="dbc-test.cloud.databricks.com",
    http_path="/sql/1.0/warehouses/abc",
    access_token="private-test-token",
    sql_statement="select bytes",
    image_column="bytes",
    image_name_column=None,
    max_rows=10,
    timeout_seconds=20,
    skip_undecodable=True,
)


def _worker_module(tmp_path, monkeypatch, source):
    (tmp_path / "isolated_fixture.py").write_text(source)
    package = Path(v1.__file__).parents[5]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), str(package)]))
    monkeypatch.setattr(isolation, "_WORKER_MODULE", "isolated_fixture")


def test_successful_child_roundtrip_leaves_parent_connector_unchanged(
    tmp_path, monkeypatch
):
    import thrift.transport.THttpClient  # noqa: F401
    from databricks.sql.auth import thrift_http_client

    original = thrift_http_client.THttpClient
    _worker_module(
        tmp_path,
        monkeypatch,
        """
import sys
import numpy as np
import cv2
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import v1, worker, transport
v1.validate_destination = lambda *a, **kw: None
v1.iter_rows = lambda *a, **kw: (item for item in [{"bytes": cv2.imencode(".png", np.full((4, 5, 3), 100, dtype=np.uint8))[1].tobytes(), "name": "sample"}])
transport.install_pinned_transport = lambda **kw: None
sys.exit(worker._main())
""",
    )

    def invoke():
        return isolation.run_isolated(
            REQUEST, limits=LIMITS, deadline=time.monotonic() + 20
        )

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))
    assert thrift_http_client.THttpClient is original
    for rows in results:
        assert rows[0]["images"].numpy_image.shape == (4, 5, 3)
        assert np.all(rows[0]["images"].numpy_image == 100)
        assert rows[0]["metadata"] == {"name": "sample"}


def test_worker_launch_failure_cleans_directory(monkeypatch):
    seen = []

    def fail(args, **kwargs):
        seen.append(Path(args[-1]))
        assert REQUEST["access_token"] not in str(args)
        raise OSError("launch failed")

    monkeypatch.setattr(isolation.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="launch failed"):
        isolation.run_isolated(REQUEST, limits=LIMITS, deadline=time.monotonic() + 20)
    assert seen and not seen[0].exists()


def test_hung_child_is_killed_reaped_and_artifacts_removed(tmp_path, monkeypatch):
    _worker_module(
        tmp_path,
        monkeypatch,
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(100)\n",
    )
    monkeypatch.setattr(isolation, "_CLEANUP_SECONDS", 0.5)
    launched = []
    real_popen = subprocess.Popen

    def record(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(isolation.subprocess, "Popen", record)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        isolation.run_isolated(REQUEST, limits=LIMITS, deadline=started + 0.3)
    assert time.monotonic() - started < 3
    assert launched[0].poll() is not None
    assert not Path(launched[0].args[-1]).exists()


def test_production_worker_rejects_host_without_exposing_token():
    with pytest.raises(ValueError, match="server_hostname") as error:
        isolation.run_isolated(
            dict(REQUEST, server_hostname="localhost"),
            limits=LIMITS,
            deadline=time.monotonic() + 20,
        )
    assert REQUEST["access_token"] not in str(error.value)


def test_expired_run_never_launches(monkeypatch):
    monkeypatch.setattr(
        isolation.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched")
    )
    with pytest.raises(TimeoutError):
        isolation.run_isolated(REQUEST, limits=LIMITS, deadline=time.monotonic() - 1)


@pytest.mark.parametrize("policy", ["disabled", "sinks", "missing"])
def test_production_guards_refuse_before_launch(monkeypatch, policy):
    monkeypatch.setattr(
        isolation.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched")
    )
    monkeypatch.setattr(v1, "block_enabled", lambda: policy != "disabled")
    monkeypatch.setattr(v1, "connector_available", lambda: policy != "missing")
    with pytest.raises(RuntimeError):
        v1.DatabricksSQLImageSourceBlockV1(disable_sinks=policy == "sinks").run(
            **REQUEST
        )


def _output(tmp_path, array, **row_changes):
    np.save(tmp_path / "0.npy", array, allow_pickle=False)
    row = dict(file="0.npy", image_names="sample", metadata={})
    row.update(row_changes)
    (tmp_path / "manifest.json").write_text(json.dumps(dict(version=1, rows=[row])))


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((1, 1, 4), dtype=np.uint8),
        np.zeros((1, 1, 3), dtype=np.float64),
        np.zeros((40, 40, 3), dtype=np.uint8),
    ],
)
def test_rejects_invalid_array_before_loading(tmp_path, monkeypatch, array):
    _output(tmp_path, array)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("allocated"))
    with pytest.raises(RuntimeError, match="array"):
        isolation._read_outputs(
            tmp_path, max_rows=10, limits=LIMITS, deadline=time.monotonic() + 20
        )


def test_rejects_path_traversal(tmp_path):
    _output(tmp_path, np.zeros((1, 1, 3), dtype=np.uint8), file="../outside.npy")
    with pytest.raises(RuntimeError, match="path"):
        isolation._read_outputs(
            tmp_path, max_rows=10, limits=LIMITS, deadline=time.monotonic() + 20
        )


def test_rejects_symlink(tmp_path):
    _output(tmp_path, np.zeros((1, 1, 3), dtype=np.uint8))
    (tmp_path / "0.npy").rename(tmp_path / "outside.npy")
    (tmp_path / "0.npy").symlink_to(tmp_path / "outside.npy")
    with pytest.raises(RuntimeError, match="symlink"):
        isolation._read_outputs(
            tmp_path, max_rows=10, limits=LIMITS, deadline=time.monotonic() + 20
        )


def test_rejects_inflated_header_before_allocation(tmp_path, monkeypatch):
    _output(tmp_path, np.zeros((1, 1, 3), dtype=np.uint8))
    with (tmp_path / "0.npy").open("wb") as stream:
        np.lib.format.write_array_header_1_0(
            stream,
            dict(descr="|u1", fortran_order=False, shape=(100000000, 100000000, 3)),
        )
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("allocated"))
    with pytest.raises(RuntimeError, match="array"):
        isolation._read_outputs(
            tmp_path, max_rows=10, limits=LIMITS, deadline=time.monotonic() + 20
        )


def test_rejects_budget_before_allocation(tmp_path, monkeypatch):
    _output(tmp_path, np.zeros((4, 5, 3), dtype=np.uint8))
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("allocated"))
    with pytest.raises(RuntimeError, match="budget"):
        isolation._read_outputs(
            tmp_path,
            max_rows=10,
            limits=dict(LIMITS, max_run_bytes=50),
            deadline=time.monotonic() + 20,
        )


def test_child_error_redacts_token(tmp_path, monkeypatch):
    _worker_module(
        tmp_path,
        monkeypatch,
        """
import sys
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import v1, worker, transport
def fail(*a, **kw):
    raise ValueError("private-test-token was rejected")
v1.validate_destination = fail
transport.install_pinned_transport = lambda **kw: None
sys.exit(worker._main())
""",
    )
    with pytest.raises(ValueError, match="was rejected") as error:
        isolation.run_isolated(REQUEST, limits=LIMITS, deadline=time.monotonic() + 20)
    assert REQUEST["access_token"] not in str(error.value)


def test_worker_crash_has_controlled_error(tmp_path, monkeypatch):
    _worker_module(tmp_path, monkeypatch, "import os\nos._exit(7)\n")
    with pytest.raises(RuntimeError, match="worker failed"):
        isolation.run_isolated(REQUEST, limits=LIMITS, deadline=time.monotonic() + 20)


def test_partial_manifest_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text('{"version": 1, "rows": [')
    with pytest.raises(RuntimeError, match="manifest"):
        isolation._read_outputs(
            tmp_path, max_rows=10, limits=LIMITS, deadline=time.monotonic() + 20
        )


def test_deadline_checked_after_last_row(monkeypatch):
    import cv2

    payload = cv2.imencode(".png", np.zeros((4, 5, 3), dtype=np.uint8))[1].tobytes()
    monkeypatch.setattr(v1, "validate_destination", lambda *a, **kw: None)
    monkeypatch.setattr(
        v1, "iter_rows", lambda *a, **kw: (item for item in [{"bytes": payload}])
    )
    original = v1.decode_image

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        time.sleep(0.05)
        return result

    monkeypatch.setattr(v1, "decode_image", delayed)
    with pytest.raises(TimeoutError):
        v1.DatabricksSQLImageSourceBlockV1()._run_in_process(
            **dict(REQUEST, max_rows=1), deadline=time.monotonic() + 0.02
        )


def test_unicode_metadata_roundtrips_within_byte_budget(tmp_path, monkeypatch):
    _worker_module(
        tmp_path,
        monkeypatch,
        """
import sys
import numpy as np
import cv2
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import v1, worker, transport
v1.validate_destination = lambda *a, **kw: None
row = {"bytes": cv2.imencode(".png", np.zeros((4, 5, 3), dtype=np.uint8))[1].tobytes()}
row.update({str(i): chr(233)*4096 for i in range(5)})
v1.iter_rows = lambda *a, **kw: (item for item in [row])
transport.install_pinned_transport = lambda **kw: None
sys.exit(worker._main())
""",
    )
    rows = isolation.run_isolated(
        dict(REQUEST, max_rows=1), limits=LIMITS, deadline=time.monotonic() + 20
    )
    assert rows[0]["metadata"] == {str(i): chr(233) * 4096 for i in range(5)}


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require Unix")
def test_fifo_is_rejected_without_blocking(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(version=1, rows=[dict(file="0.npy", image_names="n", metadata={})])
        )
    )
    os.mkfifo(tmp_path / "0.npy")
    with pytest.raises(RuntimeError, match="regular file"):
        isolation._read_outputs(
            tmp_path, max_rows=10, limits=LIMITS, deadline=time.monotonic() + 20
        )


def test_regular_outputs_work_without_unix_open_flags(tmp_path, monkeypatch):
    _output(tmp_path, np.zeros((1, 1, 3), dtype=np.uint8))
    monkeypatch.delattr(os, "O_NONBLOCK", raising=False)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    rows = isolation._read_outputs(
        tmp_path, max_rows=10, limits=LIMITS, deadline=time.monotonic() + 20
    )
    assert rows[0]["images"].shape == (1, 1, 3)
