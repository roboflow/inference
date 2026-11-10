"""Private one-invocation worker; credentials arrive only through stdin."""

import json
import signal
import sys
import time
from pathlib import Path

_MAX_REQUEST_BYTES = 1024 * 1024


def _terminate(signum, frame):
    raise TimeoutError("Databricks query exceeded timeout_seconds")


def _main() -> int:
    directory = Path(sys.argv[1])
    request = {}
    signal.signal(signal.SIGTERM, _terminate)
    try:
        payload = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
        if len(payload) > _MAX_REQUEST_BYTES:
            raise ValueError("Databricks worker request exceeds its limit")

        message = json.loads(payload)
        if message["version"] != 1:
            raise ValueError("Unsupported Databricks worker request")

        request = message["request"]
        limits = message["limits"]
        deadline = message["deadline"]
        if time.monotonic() >= deadline:
            raise TimeoutError("Databricks query exceeded timeout_seconds")

        # Imports and DNS are inside the supervised process. The standalone
        # Workflows package must suffice; no inference host is constructed here.
        import numpy as np
        from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
            v1,
        )
        from roboflow_workflows.core_steps.integrations.databricks.sql_image_source.transport import (
            install_pinned_transport,
        )

        v1.MAX_RUN_BYTES = limits["max_run_bytes"]
        v1.MAX_IMAGE_PAYLOAD_BYTES = limits["max_image_bytes"]
        v1.MAX_IMAGE_PIXELS = limits["max_image_pixels"]
        v1.ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES = limits[
            "allow_non_global"
        ]
        v1.running_on_hosted_platform = lambda: limits["hosted"]
        # Parent already enforced enablement. A spawned interpreter's host
        # defaults must not override the actual parent's configuration.
        from roboflow_workflows.core_steps.integrations.databricks import availability

        availability.DATABRICKS_SQL_IMAGE_SOURCE_ENABLED = True
        v1.block_enabled = lambda: True
        install_pinned_transport(worker_only=True)
        rows = v1.DatabricksSQLImageSourceBlockV1()._run_in_process(
            **request, deadline=deadline
        )
        manifest = []
        for index, row in enumerate(rows):
            v1._check_deadline(deadline)
            if row.get("errors") is not None:
                manifest.append({"errors": row["errors"]})
                continue

            filename = f"{index}.npy"
            np.save(directory / filename, row["images"].numpy_image, allow_pickle=False)
            manifest.append(
                dict(
                    file=filename,
                    image_names=row["image_names"],
                    metadata=row["metadata"],
                )
            )
        v1._check_deadline(deadline)
        (directory / "manifest.json").write_text(
            json.dumps(dict(version=1, rows=manifest), ensure_ascii=False),
            encoding="utf-8",
        )

        return 0
    except BaseException as error:
        # Never echo a raw traceback, query, or connector exception to logs.
        token = request.get("access_token", "")
        text = str(error)
        if token:
            text = text.replace(token, "***")
        kind = (
            "TimeoutError"
            if isinstance(error, TimeoutError)
            else "ValueError" if isinstance(error, ValueError) else "RuntimeError"
        )
        (directory / "error.json").write_text(
            json.dumps(dict(kind=kind, message=text[:256])), encoding="utf-8"
        )

        return 1


if __name__ == "__main__":
    sys.exit(_main())
