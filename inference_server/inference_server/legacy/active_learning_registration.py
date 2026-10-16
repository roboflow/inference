"""Registration of legacy route inferences in active learning."""

import asyncio
import importlib.util
import io
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from hashlib import sha256
from typing import Any, Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
from inference_model_manager.backends.decode import make_decoder

from inference_sdk.http.utils.aliases import resolve_roboflow_model_alias
from inference_server import configuration

logger = logging.getLogger(__name__)

MAX_MIDDLEWARES = 256
WORKERS = 4
MAX_PENDING_JOBS = 64
SERVERLESS_WAIT_TIMEOUT_S = 30

_NPY_MAGIC = b"\x93NUMPY"
_decode_encoded_image = make_decoder("imagecodecs", device="cpu")


@dataclass
class _Job:
    api_key: str
    model_id: str
    target_dataset: str
    prediction_type: str
    inference_id: Optional[str]
    images: List[Union[bytes, np.ndarray]]
    responses: list


@dataclass
class _Slot:
    lock: threading.Lock = field(default_factory=threading.Lock)
    middleware: Any = None
    created_at: float = 0.0
    busy: int = 0


class ActiveLearningRegistrar:
    """Owns the worker pool and the per-key middlewares of active learning."""

    def __init__(
        self,
        *,
        create_middleware: Callable[..., Any],
        max_age_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._create_middleware = create_middleware
        self._max_age_s = max_age_s
        self._clock = clock
        self._executor = ThreadPoolExecutor(
            max_workers=WORKERS, thread_name_prefix="active-learning"
        )
        self._lock = threading.Lock()
        self._slots: "OrderedDict[Tuple[str, str, str], _Slot]" = OrderedDict()
        self._pending = 0
        self.dropped = 0
        self.timed_out = 0

    def submit(self, job: _Job) -> Optional[Future]:
        """Queue a registration unless the pending bound is reached.

        Args:
            job: Registration to run on the worker pool.

        Returns:
            The future of the job, or None when the job was dropped.
        """
        with self._lock:
            if self._pending >= MAX_PENDING_JOBS:
                self.dropped += 1
                return None

            self._pending += 1

        try:
            future = self._executor.submit(self.run, job)
        except RuntimeError:
            with self._lock:
                self._pending -= 1
                self.dropped += 1
            return None

        future.add_done_callback(self._release)

        return future

    def run(self, job: _Job) -> None:
        """Register the images and predictions of one request.

        Args:
            job: Registration to run.
        """
        try:
            middleware = self._middleware_for(job)
            if not middleware.active:
                return None

            images = [_decode_image(data) for data in job.images]
            predictions = [
                response.model_dump(by_alias=True, exclude={"visualization"})
                for response in job.responses
            ]

            middleware.register_batch(
                images=images,
                predictions=predictions,
                prediction_type=job.prediction_type,
                inference_id=job.inference_id,
            )
        except Exception as error:
            logger.warning(
                "Active learning registration failed: %s", type(error).__name__
            )

    def is_known_inactive(self, job: _Job) -> bool:
        """Tell whether a live middleware of the job's key has no configuration.

        Args:
            job: Registration about to be submitted.

        Returns:
            True when a middleware that has not expired is held and is inactive.
        """
        key = self._key_of(job)
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                return False

            middleware = slot.middleware
            live = (
                middleware is not None
                and self._clock() - slot.created_at < self._max_age_s
            )
            if live:
                self._slots.move_to_end(key)

        if not live:
            return False

        return not middleware.active

    def record_timeout(self) -> None:
        """Count a request released by the serverless wait deadline."""
        with self._lock:
            self.timed_out += 1

    def shutdown(self) -> None:
        """Stop accepting jobs and cancel the queued ones."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _release(self, _: Future) -> None:
        with self._lock:
            self._pending -= 1

    @staticmethod
    def _key_of(job: _Job) -> Tuple[str, str, str]:
        return (
            sha256(job.api_key.encode("utf-8")).hexdigest(),
            job.model_id,
            job.target_dataset,
        )

    def _middleware_for(self, job: _Job) -> Any:
        key = self._key_of(job)
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = _Slot()
                self._admit(key, slot)
            else:
                self._slots.move_to_end(key)
            slot.busy += 1

        try:
            with slot.lock:
                expired = self._clock() - slot.created_at >= self._max_age_s
                if slot.middleware is None or expired:
                    slot.middleware = self._create_middleware(
                        api_key=job.api_key,
                        target_dataset=job.target_dataset,
                        model_id=job.model_id,
                    )
                    slot.created_at = self._clock()
                middleware = slot.middleware
        finally:
            with self._lock:
                slot.busy -= 1

        return middleware

    def _admit(self, key: Tuple[str, str, str], slot: _Slot) -> None:
        if len(self._slots) >= MAX_MIDDLEWARES:
            victim = next(
                (known for known, held in self._slots.items() if not held.busy),
                None,
            )
            if victim is None:
                return None
            del self._slots[victim]
        self._slots[key] = slot


class _ConfigurationFreeCache:
    """Shared cache that keeps no active learning configuration entries."""

    def __init__(self, cache: Any, configuration_prefix: str) -> None:
        self._cache = cache
        self._configuration_prefix = configuration_prefix

    def get(self, key: str) -> Any:
        if key.startswith(self._configuration_prefix):
            return None

        return self._cache.get(key)

    def set(self, key: str, *args: Any, **kwargs: Any) -> Any:
        if key.startswith(self._configuration_prefix):
            return None

        return self._cache.set(key, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cache, name)


_REGISTRAR: Optional[ActiveLearningRegistrar] = None


def start() -> Optional[ActiveLearningRegistrar]:
    """Create the registrar when active learning is enabled and available.

    Returns:
        The registrar, or None when active learning is off.
    """
    global _REGISTRAR
    if not configuration.ACTIVE_LEARNING_ENABLED:
        return None
    if not _workflows_installed():
        logger.info("Active learning disabled: roboflow-workflows is not installed")
        return None

    from inference_server.active_learning.configuration import (
        ACTIVE_LEARNING_CONFIG_CACHE_EXPIRE,
        ACTIVE_LEARNING_CONFIG_CACHE_KEY_PREFIX,
    )
    from inference_server.active_learning.middlewares import ActiveLearningMiddleware
    from inference_server.workflows import host

    _REGISTRAR = ActiveLearningRegistrar(
        create_middleware=partial(
            ActiveLearningMiddleware.init,
            cache=_ConfigurationFreeCache(
                host.WORKFLOWS_CACHE,
                ACTIVE_LEARNING_CONFIG_CACHE_KEY_PREFIX,
            ),
            platform_client=host.PLATFORM_CLIENT,
        ),
        max_age_s=ACTIVE_LEARNING_CONFIG_CACHE_EXPIRE,
    )

    return _REGISTRAR


def stop() -> None:
    """Shut the registrar down without waiting for queued jobs."""
    global _REGISTRAR
    registrar, _REGISTRAR = _REGISTRAR, None
    if registrar is not None:
        registrar.shutdown()


async def register_inference(
    inference_request: Any,
    *,
    task_type: str,
    payloads: list,
    responses: list,
    eligible: bool,
) -> None:
    """Hand a finished inference to active learning when the request allows it.

    In serverless mode the call returns after the registration finished or
    after the wait deadline; otherwise it returns as soon as the registration
    is queued.

    Args:
        inference_request: Request entity of the inference.
        task_type: Task type of the model that served the request.
        payloads: Image payloads of the request.
        responses: Repacked responses, one per payload.
        eligible: Whether the route takes part in active learning.
    """
    try:
        registrar = _REGISTRAR
        if registrar is None or not eligible:
            return None

        job = _job_for(
            inference_request,
            task_type=task_type,
            payloads=payloads,
            responses=responses,
        )
        if job is None or registrar.is_known_inactive(job):
            return None

        future = registrar.submit(job)
        if future is not None and (
            configuration.LAMBDA or configuration.GCP_SERVERLESS
        ):
            if not await _completion_of(future, SERVERLESS_WAIT_TIMEOUT_S):
                registrar.record_timeout()
    except Exception as error:
        logger.warning("Active learning registration failed: %s", type(error).__name__)


def _workflows_installed() -> bool:
    return importlib.util.find_spec("roboflow_workflows") is not None


def _job_for(
    inference_request: Any, *, task_type: str, payloads: list, responses: list
) -> Optional[_Job]:
    if getattr(inference_request, "disable_active_learning", False):
        return None
    if not inference_request.api_key:
        return None
    if not hasattr(inference_request, "active_learning_target_dataset"):
        return None

    model_id = resolve_roboflow_model_alias(inference_request.model_id)
    target_dataset = (
        inference_request.active_learning_target_dataset or model_id.split("/")[0]
    )
    job = _Job(
        api_key=inference_request.api_key,
        model_id=model_id,
        target_dataset=target_dataset,
        prediction_type=task_type,
        inference_id=inference_request.id,
        images=[payload.data for payload in payloads],
        responses=responses,
    )

    return job


async def _completion_of(future: Future, timeout_s: float) -> bool:
    loop = asyncio.get_running_loop()
    finished = asyncio.Event()

    def _notify(_: Future) -> None:
        try:
            loop.call_soon_threadsafe(finished.set)
        except RuntimeError:
            return None

    future.add_done_callback(_notify)
    try:
        await asyncio.wait_for(finished.wait(), timeout_s)
    except asyncio.TimeoutError:
        return False

    return True


def _decode_image(data: Union[bytes, np.ndarray]) -> np.ndarray:
    if isinstance(data, np.ndarray):
        image = data
    elif data[:6] == _NPY_MAGIC:
        image = np.load(io.BytesIO(data), allow_pickle=False)
    else:
        image = _decode_encoded_image(data)
    if image.ndim == 2 or image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image
