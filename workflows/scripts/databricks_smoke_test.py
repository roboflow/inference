"""Smoke test of the Databricks SQL Image Source against a real warehouse.

Not part of the unit suite: it needs a warehouse. Provide the connection in the
environment (never on the command line) and run from the ``workflows`` folder:

    export DATABRICKS_SERVER_HOSTNAME=dbc-xxxx.cloud.databricks.com
    export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxx
    export DATABRICKS_TOKEN=...            # read-only token
    export DATABRICKS_SQL="SELECT id, bytes FROM catalog.schema.images LIMIT 3"
    python scripts/databricks_smoke_test.py [--name-column id] [--timeout 120]

It compiles a Workflow with the block, runs it through the real
ExecutionEngine (so the pinned transport, limits and deadline are exercised),
and prints names, metadata and image shapes. Exit code 0 means every row came
back as an image. The token is never printed.
"""

import argparse
import json
import os
import sys
import time

from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.execution_engine.core import ExecutionEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-column", default=None)
    parser.add_argument("--image-column", default="bytes")
    parser.add_argument("--max-rows", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    missing = [
        name
        for name in (
            "DATABRICKS_SERVER_HOSTNAME",
            "DATABRICKS_HTTP_PATH",
            "DATABRICKS_TOKEN",
            "DATABRICKS_SQL",
        )
        if not os.environ.get(name)
    ]
    if missing:
        print("missing environment variables:", ", ".join(missing))
        return 2

    step = {
        "type": "roboflow_core/databricks_sql_image_source@v1",
        "name": "db",
        "server_hostname": os.environ["DATABRICKS_SERVER_HOSTNAME"],
        "http_path": os.environ["DATABRICKS_HTTP_PATH"],
        "access_token": "$inputs.token",
        "sql_statement": "$inputs.sql",
        "image_column": args.image_column,
        "max_rows": args.max_rows,
        "timeout_seconds": args.timeout,
    }
    if args.name_column:
        step["image_name_column"] = args.name_column
    workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "token"},
            {"type": "WorkflowParameter", "name": "sql"},
        ],
        "steps": [step],
        "outputs": [
            {"type": "JsonField", "name": "images", "selector": "$steps.db.images"},
            {"type": "JsonField", "name": "names", "selector": "$steps.db.image_names"},
            {"type": "JsonField", "name": "metadata", "selector": "$steps.db.metadata"},
        ],
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={"workflows_core.step_execution_mode": StepExecutionMode.LOCAL},
        max_concurrent_steps=1,
    )
    started = time.monotonic()
    result = engine.run(
        runtime_parameters={
            "token": os.environ["DATABRICKS_TOKEN"],
            "sql": os.environ["DATABRICKS_SQL"],
        }
    )[0]
    elapsed = time.monotonic() - started

    images = result["images"]
    print(f"rows: {len(images)}  elapsed: {elapsed:.1f}s")
    for image, name, metadata in zip(images, result["names"], result["metadata"]):
        shape = getattr(image, "numpy_image", None)
        shape = None if shape is None else tuple(shape.shape)
        print(f"  {name}: image={shape} metadata={json.dumps(metadata)[:200]}")
    return 0 if images else 1


if __name__ == "__main__":
    sys.exit(main())
