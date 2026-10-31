"""The frozen Windows server must survive a strict redirected console."""

import ast
import asyncio
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace


def test_startup_banner_keeps_server_running_on_cp1252_console():
    source = Path(__file__).resolve().parents[3] / "app_bundles/windows/run_inference.py"
    function = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_serve_with_banner"
    )
    served = []

    class Server:
        def __init__(self, config):
            self.started = self

        async def wait(self):
            pass

        async def serve(self):
            served.append(True)

    logger = SimpleNamespace(handlers=[], addFilter=lambda value: None)
    namespace = {
        "asyncio": asyncio,
        "os": SimpleNamespace(environ={"HOST": "127.0.0.1"}),
        "app": object(),
        "logging": SimpleNamespace(
            getLogger=lambda name: logger, StreamHandler=logging.StreamHandler
        ),
        "FilteredAccessLogConfig": lambda: object(),
        "uvicorn": SimpleNamespace(Config=lambda *args, **kwargs: kwargs, Server=Server),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    output = io.BytesIO()
    console = io.TextIOWrapper(output, encoding="cp1252", errors="strict")
    with redirect_stdout(console):
        asyncio.run(namespace["_serve_with_banner"]())
    console.flush()
    assert b"Roboflow Inference is ready" in output.getvalue()
    assert b"http://127.0.0.1:9001/" in output.getvalue()
    assert served == [True]
