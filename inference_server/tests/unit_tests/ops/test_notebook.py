import asyncio
import importlib.util
import threading
import time
from types import SimpleNamespace

import pytest

from inference_server.ops import notebook
from inference_server.ops import router as ops_router
from tests.unit_tests.ops.conftest import running_on_event_loop

AWKWARD_PASSWORD = "pa ss'w\"ord --ServerApp.token=x"


@pytest.fixture
def jupyter(monkeypatch, tmp_path):
    state = SimpleNamespace(
        running=False,
        popen_calls=[],
        processes=[],
        popen_error=None,
        probes=[],
        on_event_loop=[],
        delays=[],
        notebook_dir=tmp_path / "notebooks",
    )

    def _fake_get(url, **kwargs):
        state.probes.append((url, kwargs))
        if not state.running:
            raise ConnectionError("nothing listens")

        return SimpleNamespace(status_code=200)

    def _fake_popen(*args, **kwargs):
        state.popen_calls.append((args, kwargs))
        state.on_event_loop.append(running_on_event_loop())
        if state.popen_error is not None:
            raise state.popen_error

        process = SimpleNamespace(exit_code=None)
        process.poll = lambda: process.exit_code
        state.processes.append(process)

        return process

    async def _fake_sleep(delay: float) -> None:
        state.delays.append(delay)

    monkeypatch.setattr(notebook, "_process", None)
    monkeypatch.setattr(notebook.requests, "get", _fake_get)
    monkeypatch.setattr(notebook.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(notebook, "NOTEBOOK_DIR", state.notebook_dir)
    monkeypatch.setattr(
        ops_router,
        "asyncio",
        SimpleNamespace(sleep=_fake_sleep, to_thread=asyncio.to_thread),
    )

    return state


def test_disabled_browserless_answers_legacy_json(ops_client, jupyter):
    response = ops_client().get("/notebook/start?browserless=true")

    assert response.status_code == 200
    assert response.content == (
        b'{"success":false,"message":"Notebook server is not enabled. '
        b'Enable notebooks via the NOTEBOOK_ENABLED environment variable."}'
    )
    assert jupyter.popen_calls == []


def test_disabled_redirects_to_instructions_page(ops_client, jupyter):
    response = ops_client().get("/notebook/start")

    assert response.status_code == 307
    assert response.headers["location"] == "/notebook-instructions.html"
    assert jupyter.popen_calls == []


def test_enabled_browserless_starts_jupyter_and_reports_address(ops_client, jupyter):
    client = ops_client(
        NOTEBOOK_ENABLED=True, NOTEBOOK_PORT=9100, NOTEBOOK_PASSWORD="s3cret"
    )

    response = client.get("/notebook/start?browserless=true")

    assert response.status_code == 200
    assert response.content == (
        b'{"success":true,"message":"Jupyter Lab server started at '
        b'http://localhost:9100?token=s3cret"}'
    )
    assert jupyter.popen_calls == [
        (
            (
                [
                    "jupyter-lab",
                    "--allow-root",
                    "--port=9100",
                    "--ip=0.0.0.0",
                    f"--notebook-dir={jupyter.notebook_dir}",
                    "--NotebookApp.token=s3cret",
                    "--NotebookApp.password=s3cret",
                ],
            ),
            {},
        )
    ]
    assert jupyter.on_event_loop == [False]
    assert jupyter.notebook_dir.is_dir()
    assert jupyter.probes[0][0] == "http://localhost:9100/"


def test_enabled_redirects_to_quickstart_notebook(ops_client, jupyter):
    client = ops_client(NOTEBOOK_ENABLED=True, NOTEBOOK_PASSWORD="s3cret")

    response = client.get("/notebook/start")

    assert response.status_code == 307
    assert response.headers["location"] == (
        "http://localhost:9002/lab/tree/quickstart.ipynb?token=s3cret"
    )
    assert len(jupyter.popen_calls) == 1


def test_only_the_redirect_waits_two_seconds_and_it_does_not_block_the_loop(
    ops_client, jupyter
):
    client = ops_client(NOTEBOOK_ENABLED=True, NOTEBOOK_PASSWORD="s3cret")

    client.get("/notebook/start?browserless=true")

    assert jupyter.delays == []

    client.get("/notebook/start")

    assert jupyter.delays == [2.0]


def test_running_notebook_is_not_started_again(ops_client, jupyter):
    jupyter.running = True
    client = ops_client(NOTEBOOK_ENABLED=True, NOTEBOOK_PASSWORD="s3cret")

    response = client.get("/notebook/start?browserless=true")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert jupyter.popen_calls == []


def test_missing_jupyter_binary_answers_fixed_500(ops_client, jupyter):
    jupyter.popen_error = FileNotFoundError(2, "No such file", "jupyter-lab")
    client = ops_client(NOTEBOOK_ENABLED=True, NOTEBOOK_PASSWORD="s3cret")

    for path in ("/notebook/start", "/notebook/start?browserless=true"):
        response = client.get(path)

        assert response.status_code == 500
        assert response.content == (
            b'{"message":"Notebook server could not be started. '
            b'Check that jupyter-lab is installed in this environment."}'
        )
        assert "s3cret" not in response.text


def test_unset_password_uses_a_token_generated_for_this_process(ops_client, jupyter):
    client = ops_client(NOTEBOOK_ENABLED=True, NOTEBOOK_PASSWORD=None)

    first = client.get("/notebook/start?browserless=true")
    jupyter.processes[0].exit_code = 0
    second = client.get("/notebook/start")

    token = notebook.notebook_token()
    assert len(token) >= 32
    assert token != "roboflow"
    assert first.json()["message"].endswith(f"?token={token}")
    assert second.headers["location"].endswith(f"?token={token}")
    assert [call[0][0][5:] for call in jupyter.popen_calls] == [
        [f"--NotebookApp.token={token}", f"--NotebookApp.password={token}"]
    ] * 2


def test_generated_token_differs_between_processes():
    spec = importlib.util.spec_from_file_location("notebook_copy", notebook.__file__)
    second_process_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(second_process_module)

    assert second_process_module._GENERATED_TOKEN != notebook._GENERATED_TOKEN


def test_empty_password_counts_as_unset(monkeypatch):
    monkeypatch.setattr(notebook.configuration, "NOTEBOOK_PASSWORD", "")

    assert notebook.notebook_token() == notebook._GENERATED_TOKEN


def test_awkward_password_reaches_jupyter_as_one_argument(ops_client, jupyter):
    client = ops_client(NOTEBOOK_ENABLED=True, NOTEBOOK_PASSWORD=AWKWARD_PASSWORD)

    response = client.get("/notebook/start?browserless=true")

    assert response.status_code == 200
    (command,), kwargs = jupyter.popen_calls[0]
    assert kwargs == {}
    assert command == [
        "jupyter-lab",
        "--allow-root",
        "--port=9002",
        "--ip=0.0.0.0",
        f"--notebook-dir={jupyter.notebook_dir}",
        f"--NotebookApp.token={AWKWARD_PASSWORD}",
        f"--NotebookApp.password={AWKWARD_PASSWORD}",
    ]


def test_token_is_percent_encoded_in_both_urls(ops_client, jupyter):
    password = "pa ss&w#o'rd\""
    client = ops_client(NOTEBOOK_ENABLED=True, NOTEBOOK_PASSWORD=password)
    encoded = "pa%20ss%26w%23o%27rd%22"

    browserless = client.get("/notebook/start?browserless=true")
    jupyter.processes[0].exit_code = 0
    redirect = client.get("/notebook/start")

    assert browserless.json()["message"].endswith(
        f"http://localhost:9002?token={encoded}"
    )
    assert redirect.headers["location"] == (
        f"http://localhost:9002/lab/tree/quickstart.ipynb?token={encoded}"
    )


def test_generated_token_and_plain_value_are_not_altered_by_encoding(
    ops_client, jupyter
):
    client = ops_client(NOTEBOOK_ENABLED=True, NOTEBOOK_PASSWORD="roboflow")
    plain = client.get("/notebook/start?browserless=true")
    assert plain.json()["message"].endswith("?token=roboflow")

    jupyter.processes[0].exit_code = 0
    client = ops_client(NOTEBOOK_ENABLED=True, NOTEBOOK_PASSWORD=None)
    generated = client.get("/notebook/start?browserless=true")
    assert generated.json()["message"].endswith(f"?token={notebook._GENERATED_TOKEN}")


def test_concurrent_starts_launch_one_server(monkeypatch, jupyter):
    def _slow_probe(url, **kwargs):
        time.sleep(0.05)
        raise ConnectionError("nothing listens")

    monkeypatch.setattr(notebook.requests, "get", _slow_probe)
    barrier = threading.Barrier(2)

    def _start() -> None:
        barrier.wait()
        notebook.start_notebook()

    threads = [threading.Thread(target=_start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(jupyter.popen_calls) == 1


def test_live_process_blocks_a_second_launch_even_when_probe_stays_silent(jupyter):
    notebook.start_notebook()
    notebook.start_notebook()

    assert len(jupyter.popen_calls) == 1


def test_exited_process_allows_a_new_launch(jupyter):
    notebook.start_notebook()
    jupyter.processes[0].exit_code = 1
    notebook.start_notebook()

    assert len(jupyter.popen_calls) == 2
