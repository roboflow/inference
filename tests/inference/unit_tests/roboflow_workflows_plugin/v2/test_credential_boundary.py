"""Credential diagnostics: inspect records before handlers or formatting."""

import concurrent.futures
import io
import json
import logging
import logging.handlers
import queue
import threading
import traceback
from urllib.parse import quote, quote_plus

import pytest
import structlog
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.compilation import WorkflowReference

from inference.core import exceptions
from inference.roboflow_workflows_plugin.v2 import credential_boundary as boundary
from inference.roboflow_workflows_plugin.v2 import platform_adapter, workflow_resolver

KEY = "qa_ONLY_long_key_A/b+c?= space"
VARIANTS = (KEY, quote(KEY, safe=""), quote_plus(KEY), "qa_O***", "qa***ce")


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def assert_safe(error):
    pending, seen = [error], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        text = str(item) + repr(item) + repr(vars(item))
        assert all(value not in text for value in VARIANTS)
        for child in (item.__cause__, item.__context__, *vars(item).values()):
            if isinstance(child, BaseException):
                pending.append(child)
    assert all(
        value not in "".join(traceback.format_exception(error)) for value in VARIANTS
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "operation", ("workspace", "metadata", "search", "image", "resolver")
)
def test_all_host_operations_use_fixed_errors(monkeypatch, variant, operation):
    class UnsafeError(Exception):
        pass

    original = UnsafeError("provider api_key=" + variant)
    original.request = {"url": variant}

    def fail(**kwargs):
        raise original

    adapter = platform_adapter.RoboflowPlatformAdapter()
    key = Secret(KEY, name="QA_ONLY")
    if operation == "workspace":
        monkeypatch.setattr(
            platform_adapter.roboflow_api, "get_roboflow_workspace", fail
        )
        invoke = lambda: adapter.get_workspace(api_key=key)
    elif operation == "metadata":
        monkeypatch.setattr(platform_adapter.roboflow_api, "add_custom_metadata", fail)
        invoke = lambda: adapter.add_custom_metadata(
            api_key=key,
            workspace_id="ws",
            inference_ids=["id"],
            field_name="f",
            field_value="v",
        )
    elif operation == "search":
        monkeypatch.setattr(
            platform_adapter.roboflow_api, "search_project_images_at_roboflow", fail
        )
        invoke = lambda: adapter.search_project_images(
            api_key=key, workspace="ws", project="p", image_base64="A", limit=1
        )
    elif operation == "image":
        monkeypatch.setattr(platform_adapter.image_utils, "load_image_from_url", fail)
        invoke = lambda: adapter.load_candidate_image(
            "http://127.0.0.1/?api_key=" + variant
        )
    else:
        monkeypatch.setattr(
            workflow_resolver.roboflow_api, "get_workflow_specification", fail
        )
        resolver = workflow_resolver.RoboflowWorkflowResolver(api_key=KEY)
        invoke = lambda: resolver(WorkflowReference(workspace_id="ws", workflow_id="p"))
    with pytest.raises(
        RuntimeError if operation == "resolver" else PlatformRequestError
    ) as info:
        invoke()
    assert info.value.__cause__ is info.value.__context__ is None
    assert_safe(info.value)
    if operation != "resolver":
        assert info.value.error_type == "builtins.RuntimeError"


def test_safe_error_even_when_caller_is_handling_unsafe_exception():
    try:
        raise RuntimeError(KEY)
    except RuntimeError:

        def fail():
            raise ValueError(quote_plus(KEY))

        with pytest.raises(ValueError) as info:
            boundary.call_host(boundary.HostOperation.WORKFLOW, fail)
        assert info.value.__context__ is None
        assert_safe(info.value)


@pytest.mark.parametrize(
    "error_class",
    [
        exceptions.RoboflowAPINotAuthorizedError,
        exceptions.RoboflowAPIForbiddenError,
        exceptions.RoboflowAPINotNotFoundError,
        exceptions.RoboflowAPITimeoutError,
        exceptions.RoboflowAPIConnectionError,
        exceptions.RoboflowAPIUnsuccessfulRequestError,
        exceptions.MalformedWorkflowResponseError,
        exceptions.MalformedRoboflowAPIResponseError,
        exceptions.CacheUnavailableError,
        FileNotFoundError,
        ValueError,
    ],
)
def test_resolver_retains_fresh_expected_category(error_class):
    original = error_class(KEY)
    original.response = {"key": KEY}

    def fail():
        raise original

    with pytest.raises(error_class) as info:
        boundary.call_host(boundary.HostOperation.WORKFLOW, fail)
    assert info.value is not original
    assert vars(info.value) == {}
    assert info.value.__cause__ is info.value.__context__ is None
    assert_safe(info.value)


def test_direct_propagated_and_delayed_records_clear_all_diagnostic_payloads():
    boundary.register_diagnostic_filters()
    logger = logging.getLogger("urllib3.connectionpool")
    root = logging.getLogger()
    direct, propagated = Capture(), Capture()
    messages = queue.Queue()
    queued = logging.handlers.QueueHandler(messages)
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    logger.addHandler(direct)
    logger.addHandler(queued)
    root.addHandler(propagated)
    try:
        # exc_text is a standard reserved field, so emit a prepared record to
        # model a cached traceback without relying on Logger's extra validation.
        def prepared():
            record = logger.makeRecord(
                logger.name,
                logging.ERROR,
                __file__,
                1,
                "URL %s",
                (KEY,),
                (RuntimeError, RuntimeError(KEY), None),
            )
            record.exc_text = KEY
            record.stack_info = KEY
            record.message = KEY
            record.request = {"url": quote_plus(KEY)}
            logger.handle(record)

        boundary.call_host(boundary.HostOperation.SEARCH, prepared)
        assert len(direct.records) == len(propagated.records) == 1
        record = direct.records[0]
        assert record is propagated.records[0]
        assert record.levelno == logging.ERROR
        assert (
            record.args == ()
            and record.exc_info is record.exc_text is record.stack_info is None
        )
        assert all(value not in repr(vars(record)) for value in VARIANTS)
        delayed = messages.get_nowait()
        assert "project image search" in logging.Formatter().format(delayed)
        assert all(value not in repr(vars(delayed)) for value in VARIANTS)
    finally:
        logger.removeHandler(direct)
        logger.removeHandler(queued)
        root.removeHandler(propagated)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def test_outside_scope_is_exact_record_noop():
    boundary.register_diagnostic_filters()
    logger = logging.getLogger("urllib3.connectionpool")
    record = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        "URL %s",
        (KEY,),
        None,
        extra={"payload": {"url": quote_plus(KEY)}},
    )
    before = vars(record).copy()
    assert boundary._FILTER.filter(record)
    assert vars(record) == before
    assert record.getMessage() == "URL " + KEY


def test_concurrent_registration_nested_scopes_and_failure_cleanup():
    before = {
        name: (
            logging.getLogger(name).level,
            tuple(logging.getLogger(name).handlers),
            logging.getLogger(name).propagate,
        )
        for name in boundary.AUDITED_LOGGERS
    }

    def construct(_):
        platform_adapter.RoboflowPlatformAdapter()
        workflow_resolver.RoboflowWorkflowResolver(api_key=KEY)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(construct, range(64)))
    for name in boundary.AUDITED_LOGGERS:
        logger = logging.getLogger(name)
        assert logger.filters.count(boundary._FILTER) == 1
        assert (logger.level, tuple(logger.handlers), logger.propagate) == before[name]
    markers = []

    def inner():
        markers.append(boundary._OPERATION.get())
        raise RuntimeError(KEY)

    def outer():
        markers.append(boundary._OPERATION.get())
        with pytest.raises(RuntimeError):
            boundary.call_host(boundary.HostOperation.IMAGE, inner)
        markers.append(boundary._OPERATION.get())

    boundary.call_host(boundary.HostOperation.SEARCH, outer)
    assert markers == [
        boundary.HostOperation.SEARCH,
        boundary.HostOperation.IMAGE,
        boundary.HostOperation.SEARCH,
    ]
    assert boundary._OPERATION.get() is None
    assert boundary.call_host(boundary.HostOperation.WORKSPACE, lambda: 42) == 42
    assert boundary._OPERATION.get() is None


def test_barrier_overlaps_two_v2_keys_with_unchanged_v1_record():
    barrier = threading.Barrier(3)
    logger = logging.getLogger("urllib3.connectionpool")
    capture = Capture()
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(capture)
    keys = [KEY, "qa_SECOND_reserved/?+ space", "qa_V1_stays_visible/?+ space"]

    def host(key):
        barrier.wait(timeout=10)
        logger.debug("GET %s", quote_plus(key), extra={"wire_url": key})
        return key

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            calls = [
                executor.submit(
                    boundary.call_host, boundary.HostOperation.SEARCH, host, key=key
                )
                for key in keys[:2]
            ]
            calls.append(executor.submit(host, keys[2]))
            assert [future.result(timeout=15) for future in calls] == keys
        assert len(capture.records) == 3
        (v1,) = [
            record for record in capture.records if not hasattr(record, "v2_operation")
        ]
        assert v1.msg == "GET %s" and v1.args == (quote_plus(keys[2]),)
        assert v1.wire_url == keys[2]
        for record in capture.records:
            if record is not v1:
                assert all(
                    key not in repr(vars(record))
                    and quote_plus(key) not in repr(vars(record))
                    for key in keys
                )
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)


def test_structlog_event_dict_remains_compatible_with_processorformatter():
    logger = logging.getLogger("inference")
    capture = Capture()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ]
        )
    )
    previous = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers = [capture, handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Explicit per-logger processor chain exercises the installed structlog
    # wrapper without changing process-wide structlog configuration.
    wrapped = structlog.wrap_logger(
        logger,
        processors=[
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    try:

        def emit():
            try:
                raise RuntimeError(KEY)
            except RuntimeError:
                wrapped.error(
                    KEY, url=quote_plus(KEY), nested={"key": KEY}, exc_info=True
                )

        boundary.call_host(boundary.HostOperation.WORKFLOW, emit)
        payload = json.loads(stream.getvalue())
        assert payload == {"event": "V2 host diagnostic: saved workflow retrieval"}
        assert all(value not in repr(vars(capture.records[0])) for value in VARIANTS)
        wrapped.warning("outside", url=KEY)
        assert json.loads(stream.getvalue().splitlines()[1])["url"] == KEY
    finally:
        logger.handlers, logger.level, logger.propagate = previous


def test_structured_queue_can_delay_processorformatting():
    logger = logging.getLogger("inference")
    messages = queue.Queue()
    handler = logging.handlers.QueueHandler(messages)
    previous = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    wrapped = structlog.wrap_logger(
        logger,
        processors=[structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    try:
        boundary.call_host(
            boundary.HostOperation.WORKFLOW,
            lambda: wrapped.warning(KEY, payload={"key": quote_plus(KEY)}),
        )
        record = messages.get_nowait()
        assert all(value not in repr(vars(record)) for value in VARIANTS)
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ]
        )
        payload = json.loads(formatter.format(record))
        assert "saved workflow retrieval" in payload["event"]
    finally:
        logger.handlers, logger.level, logger.propagate = previous
