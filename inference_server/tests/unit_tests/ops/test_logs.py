import logging
import random
import re
import sys
import time
from datetime import datetime, timezone

import pytest

from inference_server import configuration, gateway, logging_config
from inference_server.ops import memory_logs

RECORD_KEYS = ["timestamp", "level", "logger", "message", "module", "line"]
WATCHED_LOGGERS = (
    None,
    logging_config.APPLICATION_LOGGER_NAME,
    logging_config.UVICORN_LOGGER_NAME,
    logging_config.UVICORN_ERROR_LOGGER_NAME,
    logging_config.UVICORN_ACCESS_LOGGER_NAME,
    "roboflow_workflows",
)


@pytest.fixture(autouse=True)
def _isolated_logging():
    saved = {
        name: (
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
        )
        for name in WATCHED_LOGGERS
    }
    memory_logs._log_entries.clear()
    yield
    memory_logs._log_entries.clear()
    for name, (handlers, level, propagate) in saved.items():
        target = logging.getLogger(name)
        target.handlers = handlers
        target.setLevel(level)
        target.propagate = propagate


def _store(
    message: str = "hello",
    *,
    level: int = logging.INFO,
    created: float = None,
    name: str = "tests.logs",
) -> None:
    record = logging.LogRecord(name, level, "/src/mod.py", 12, message, None, None)
    if created is not None:
        record.created = created
    memory_logs._memory_handler.emit(record)


def _attached(name: str = None) -> bool:
    return memory_logs._memory_handler in logging.getLogger(name).handlers


def test_route_answers_legacy_404_when_flag_is_off(ops_client):
    _store()

    response = ops_client().get("/logs")

    assert response.status_code == 404
    assert response.content == b'{"detail":"Logs endpoint not available"}'


def test_record_carries_the_six_legacy_keys(ops_client):
    created = time.time()
    _store("disk is full", level=logging.WARNING, created=created)

    response = ops_client(ENABLE_IN_MEMORY_LOGS=True).get("/logs")

    body = response.json()
    assert response.status_code == 200
    assert list(body) == ["logs", "total_count"]
    assert body["total_count"] == 1
    assert list(body["logs"][0]) == RECORD_KEYS
    assert body["logs"][0]["timestamp"] == datetime.fromtimestamp(created).isoformat()
    assert body["logs"][0]["level"] == "WARNING"
    assert body["logs"][0]["logger"] == "tests.logs"
    assert body["logs"][0]["message"].endswith(" WARNING tests.logs: disk is full")
    assert body["logs"][0]["module"] == "mod"
    assert body["logs"][0]["line"] == 12


@pytest.mark.parametrize("query", ["", "?limit=0"])
def test_default_and_zero_limit_return_the_last_hundred(ops_client, query):
    for index in range(150):
        _store(f"entry-{index}")

    body = ops_client(ENABLE_IN_MEMORY_LOGS=True).get(f"/logs{query}").json()

    assert body["total_count"] == 100
    assert body["logs"][0]["message"].endswith("entry-50")
    assert body["logs"][-1]["message"].endswith("entry-149")


def test_explicit_limit_returns_the_tail(ops_client):
    for index in range(10):
        _store(f"entry-{index}")

    body = ops_client(ENABLE_IN_MEMORY_LOGS=True).get("/logs?limit=3").json()

    assert [log["message"][-7:] for log in body["logs"]] == [
        "entry-7",
        "entry-8",
        "entry-9",
    ]
    assert body["total_count"] == 3


def test_level_filter_matches_case_insensitively(ops_client):
    _store("fine", level=logging.INFO)
    _store("broken", level=logging.ERROR)

    body = ops_client(ENABLE_IN_MEMORY_LOGS=True).get("/logs?level=error").json()

    assert [log["level"] for log in body["logs"]] == ["ERROR"]
    assert body["total_count"] == 1


def test_since_filters_with_a_naive_timestamp(ops_client):
    now = time.time()
    _store("old", created=now - 120)
    _store("new", created=now)
    since = datetime.fromtimestamp(now - 60).isoformat()

    body = (
        ops_client(ENABLE_IN_MEMORY_LOGS=True)
        .get("/logs", params={"since": since})
        .json()
    )

    assert [log["message"][-3:] for log in body["logs"]] == ["new"]


@pytest.mark.parametrize("suffix", ["Z", "+00:00"])
def test_since_filters_with_a_timezone_aware_timestamp(ops_client, suffix):
    now = time.time()
    _store("old", created=now - 120)
    _store("new", created=now)
    since = (
        datetime.fromtimestamp(now - 60, tz=timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
        + suffix
    )

    response = ops_client(ENABLE_IN_MEMORY_LOGS=True).get(
        "/logs", params={"since": since}
    )

    assert response.status_code == 200
    assert [log["message"][-3:] for log in response.json()["logs"]] == ["new"]


def test_unparsable_since_is_ignored(ops_client):
    _store("old", created=time.time() - 120)
    _store("new")

    body = (
        ops_client(ENABLE_IN_MEMORY_LOGS=True)
        .get("/logs", params={"since": "yesterday"})
        .json()
    )

    assert body["total_count"] == 2


def test_buffer_keeps_only_the_last_thousand_records(ops_client):
    for index in range(1100):
        _store(f"entry-{index}")

    body = ops_client(ENABLE_IN_MEMORY_LOGS=True).get("/logs?limit=5000").json()

    assert body["total_count"] == 1000
    assert body["logs"][0]["message"].endswith("entry-100")
    assert memory_logs._log_entries.maxlen == 1000


def test_access_line_is_stored_without_the_api_key(monkeypatch):
    monkeypatch.setattr(configuration, "ENABLE_IN_MEMORY_LOGS", True)
    logging_config.configure_logging()
    memory_logs.setup_memory_logging()

    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:51234",
        "GET",
        "/infer/object_detection?api_key=SECRET123&x=1",
        "1.1",
        200,
    )

    (entry,) = memory_logs.get_recent_logs()
    assert entry["logger"] == "uvicorn.access"
    assert "SECRET123" not in entry["message"]
    assert "/infer/object_detection?api_key=***&x=1" in entry["message"]


def test_exception_text_is_redacted_too():
    try:
        raise RuntimeError("call failed for api_key=SECRET123")
    except RuntimeError:
        record = logging.LogRecord(
            "tests.logs", logging.ERROR, "/src/mod.py", 1, "boom", None, sys.exc_info()
        )
    memory_logs._memory_handler.emit(record)

    (entry,) = memory_logs.get_recent_logs()
    assert "SECRET123" not in entry["message"]
    assert "RuntimeError" in entry["message"]


def test_nothing_is_captured_when_flag_is_off(monkeypatch):
    monkeypatch.setattr(configuration, "ENABLE_IN_MEMORY_LOGS", False)
    logging_config.configure_logging()

    memory_logs.setup_memory_logging()
    logging.getLogger("inference_server.somewhere").error("not stored")

    assert not _attached()
    assert not _attached(logging_config.APPLICATION_LOGGER_NAME)
    assert not _attached(logging_config.UVICORN_ACCESS_LOGGER_NAME)
    assert memory_logs.get_recent_logs() == []


def test_application_and_access_lines_reach_the_buffer(monkeypatch):
    monkeypatch.setattr(configuration, "ENABLE_IN_MEMORY_LOGS", True)
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)
    monkeypatch.setattr(configuration, "STRUCTURED_API_LOGGING", False)
    logging_config.configure_logging()

    memory_logs.setup_memory_logging()
    logging.getLogger("inference_server.somewhere").error("application line")
    logging.getLogger("some_library").error("library line")
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1:1", "GET", "/info", "1.1", 200
    )

    stored = [(log["logger"], log["level"]) for log in memory_logs.get_recent_logs()]
    assert stored == [
        ("inference_server.somewhere", "ERROR"),
        ("some_library", "ERROR"),
        ("uvicorn.access", "INFO"),
    ]


def test_handler_survives_repeated_logging_setup_without_duplicates(monkeypatch):
    monkeypatch.setattr(configuration, "ENABLE_IN_MEMORY_LOGS", True)
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)
    monkeypatch.setattr(configuration, "STRUCTURED_API_LOGGING", False)

    for _ in range(2):
        logging_config.configure_logging()
        memory_logs.setup_memory_logging()
    logging_config.configure_logging()

    for name in (
        None,
        logging_config.APPLICATION_LOGGER_NAME,
        logging_config.UVICORN_ACCESS_LOGGER_NAME,
    ):
        handlers = logging.getLogger(name).handlers
        assert handlers.count(memory_logs._memory_handler) == 1
    for name in (
        logging_config.APPLICATION_LOGGER_NAME,
        logging_config.UVICORN_ACCESS_LOGGER_NAME,
    ):
        owned = [
            handler
            for handler in logging.getLogger(name).handlers
            if isinstance(handler, logging_config._OwnedStreamHandler)
        ]
        assert len(owned) == 1

    logging.getLogger("inference_server.somewhere").error("once")

    assert len(memory_logs.get_recent_logs()) == 1


def test_propagating_package_logger_is_not_captured_twice(monkeypatch):
    monkeypatch.setattr(configuration, "ENABLE_IN_MEMORY_LOGS", True)
    logging_config.configure_logging()
    workflows_logger = logging.getLogger("roboflow_workflows")
    workflows_logger.propagate = True

    memory_logs.setup_memory_logging()
    workflows_logger.error("once")

    assert not _attached("roboflow_workflows")
    assert len(memory_logs.get_recent_logs()) == 1


def test_non_propagating_package_logger_is_captured(monkeypatch):
    monkeypatch.setattr(configuration, "ENABLE_IN_MEMORY_LOGS", True)
    logging_config.configure_logging()
    workflows_logger = logging.getLogger("roboflow_workflows")
    workflows_logger.propagate = False

    memory_logs.setup_memory_logging()
    workflows_logger.error("once")

    assert _attached("roboflow_workflows")
    assert len(memory_logs.get_recent_logs()) == 1


def test_structured_access_log_replaces_the_uvicorn_access_capture(monkeypatch):
    monkeypatch.setattr(configuration, "ENABLE_IN_MEMORY_LOGS", True)
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", True)
    monkeypatch.setattr(configuration, "STRUCTURED_API_LOGGING", True)
    monkeypatch.setattr(configuration, "LOG_LEVEL", "INFO")
    logging_config.configure_logging()

    memory_logs.setup_memory_logging()
    logging.getLogger(logging_config.ACCESS_LOGGER_NAME).info(
        '127.0.0.1:1 - "GET /infer/object_detection HTTP/1.1" 200'
    )

    assert not _attached(logging_config.UVICORN_ACCESS_LOGGER_NAME)
    assert [log["logger"] for log in memory_logs.get_recent_logs()] == [
        logging_config.ACCESS_LOGGER_NAME
    ]


@pytest.mark.parametrize(
    "message,expected",
    [
        ("GET /infer/x?api_key=SECRET123&x=1", "GET /infer/x?api_key=***&x=1"),
        ("GET /infer/x?api%5Fkey=SECRET123&x=1", "GET /infer/x?api_key=***&x=1"),
        ("GET /infer/x?api%255Fkey=SECRET123&x=1", "GET /infer/x?api_key=***&x=1"),
        ("GET /infer/x?api_key%3DSECRET123&x=1", "GET /infer/x?api_key=***&x=1"),
        (
            '{"Authorization": "Bearer SECRET123"}',
            '{"Authorization": "***"}',
        ),
        (
            "{'authorization': 'Bearer SECRET123'}",
            "{'authorization': '***'}",
        ),
    ],
)
def test_encoded_and_quoted_secrets_are_not_stored(message, expected):
    _store(message)

    (entry,) = memory_logs.get_recent_logs()
    assert "SECRET123" not in entry["message"]
    assert entry["message"].split("tests.logs: ", 1)[1] == expected


def test_since_compares_instants_in_a_repeated_clock_hour(ops_client, monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("tzset is not available")
    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    try:
        first = datetime(2026, 10, 25, 0, 30, tzinfo=timezone.utc).timestamp()
        second = datetime(2026, 10, 25, 1, 15, tzinfo=timezone.utc).timestamp()
        _store("first", created=first)
        _store("second", created=second)

        body = (
            ops_client(ENABLE_IN_MEMORY_LOGS=True)
            .get("/logs", params={"since": "2026-10-25T00:45:00Z"})
            .json()
        )
    finally:
        monkeypatch.undo()
        time.tzset()

    assert [log["message"][-6:] for log in body["logs"]] == ["second"]
    assert all(list(log) == RECORD_KEYS for log in body["logs"])


_UNANCHORED_URL = re.compile(r"""([a-zA-Z][a-zA-Z0-9+.-]*)://([^\s"<>]*)""")
_URL_SAMPLES = [
    "https://api.roboflow.com/?api_key=SECRET&nocache=true",
    "GET http://h/x?api_key=SECRET HTTP/1.1",
    "1abc://h/p?api_key=x",
    "xhttps://h/a and ftp://u:p@h:21/f",
    "+.-1a://h/x",
    "123://h/x https://h",
    '"https://h/x?k=1" <a href="http://h/y?token=t">',
    "url=https://gw/proxy?url=https%3A%2F%2Fapi%2F%3Fapi_key%3DSECRET",
    "https://a/x?next=http://b/y?api_key=SECRET",
    "{'url': 'https://h/p?api_key=SECRET'}",
    "a-b.c+d://x/y a://b://c",
    "://h/x a1://h 9://h",
]


def _random_samples():
    rng = random.Random(1234)
    alphabet = "ab1+.-:/ \"'<>?=&@#%xY9"
    return [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 48)))
        for _ in range(5000)
    ]


@pytest.mark.parametrize("text", _URL_SAMPLES + _random_samples())
def test_url_pattern_matches_the_unanchored_spans(text):
    unanchored = [(m.start(1), m.end()) for m in _UNANCHORED_URL.finditer(text)]
    anchored = [(m.start(2), m.end()) for m in gateway._URL.finditer(text)]

    assert anchored == unanchored


@pytest.mark.parametrize("text", _URL_SAMPLES)
def test_url_redaction_keeps_text_before_the_scheme(text):
    def _unanchored_redaction(match):
        rest = match.group(2)
        authority = re.split(r"[/?#]", rest, maxsplit=1)[0]
        if rest in (authority, authority + "/"):
            return match.group(0)
        host = re.sub(r":\d*$", "", authority.rpartition("@")[2])
        return f"{match.group(1)}://{host}/***"

    expected = _UNANCHORED_URL.sub(_unanchored_redaction, text)

    assert gateway._URL.sub(gateway._redact_url, text) == expected


@pytest.mark.parametrize("filler", ["a", "a1", "a-", "1a"])
def test_long_scheme_like_run_is_sanitized_in_linear_time(filler):
    message = "Invalid X-Request-ID: " + filler * (65536 // len(filler))

    started = time.perf_counter()
    sanitized = memory_logs._sanitize(message)
    elapsed = time.perf_counter() - started

    assert sanitized == message
    assert elapsed < 0.5
