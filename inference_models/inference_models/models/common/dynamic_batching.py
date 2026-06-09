"""
Dynamic micro-batching for HF VLM wrappers.

Each VLM HF wrapper serializes inference with a `threading.Lock` around
`self._model.generate(...)`. Under concurrent HTTP traffic all requests for
the same model queue on that lock and run one-by-one. The `DynamicBatcher`
defined here lets concurrent requests for the same model instance run as a
single batched `generate()` call, while keeping the per-request surface
(inputs in, trimmed new-token tensor out) unchanged.

The feature is opt-in via `INFERENCE_MODELS_DYNAMIC_BATCHING_ENABLED` - when
the flag is off, wrappers keep the existing lock-serialized path and no
batcher thread is ever created.
"""

import inspect
import queue
import time
import weakref
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from inference_models import configuration
from inference_models.logger import LOGGER


class InputsCollationError(Exception):
    pass


@dataclass
class _PendingRequest:
    inputs: dict
    gen_kwargs: dict
    future: Future
    enqueued_at: float = field(default_factory=time.monotonic)

    @property
    def rows(self) -> int:
        return self.inputs["input_ids"].shape[0]


class DynamicBatcher:
    """
    Per-model-instance dynamic micro-batcher.

    Requests are enqueued by `submit(...)` and consumed by a single daemon
    thread. The thread blocks for the first request, then keeps collecting
    requests until `max_batch_size` is reached or `max_wait_ms` elapsed.
    Collected requests are grouped by generation-parameter signature
    (everything except `max_new_tokens`) - requests with different sampling
    parameters never share a batch. Each group is collated into one batched
    `generate()` call running at `max(max_new_tokens)`; per-request outputs
    are then split back, trimmed to the request's own `max_new_tokens` and
    stripped of trailing padding so the output contract matches the
    single-request path exactly.

    `runner` and `collate_inputs` are injected callables, which keeps the
    batching loop unit-testable without real torch models:
    * `runner(inputs, gen_kwargs)` must return the trimmed new-token tensor
      (the `generation[:, input_len:]` equivalent) - with left-padding the
      padded input length is uniform across the batch, so the existing slice
      semantics of the wrappers carry over.
    * `collate_inputs(inputs_list)` must combine per-request inputs into one
      batch (see `collate_left_padded_inputs`).

    Failure isolation: if collation or the batched generation fails, every
    member of the group is re-run serially, so only genuinely-bad requests
    receive the exception.

    Bound-method callables are held through weak references, so the batcher
    thread never keeps an evicted model wrapper (and its weights) alive - it
    exits once the owning wrapper is garbage-collected.
    """

    # How often the idle batcher thread checks whether its owner still exists.
    _IDLE_LIVENESS_INTERVAL_S = 10.0

    def __init__(
        self,
        runner: Callable[[dict, dict], torch.Tensor],
        collate_inputs: Callable[[List[dict]], dict],
        max_batch_size: int,
        max_wait_ms: float,
        result_timeout_s: float,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        name: str = "model",
    ):
        self._runner_ref = _make_callable_ref(runner)
        self._collate_inputs_ref = _make_callable_ref(collate_inputs)
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_wait_s = max(float(max_wait_ms), 0.0) / 1000
        self._result_timeout_s = result_timeout_s
        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id
        self._name = name
        self._queue: "queue.SimpleQueue[_PendingRequest]" = queue.SimpleQueue()
        self.stats: Dict[str, int] = {
            "batches": 0,
            "requests": 0,
            "batched_requests": 0,
            "serial_fallbacks": 0,
            "max_batch_size_seen": 0,
        }
        self._thread = Thread(
            target=self._batching_loop,
            name=f"{name}-dynamic-batcher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, inputs: dict, gen_kwargs: dict) -> torch.Tensor:
        future: Future = Future()
        self._queue.put(
            _PendingRequest(inputs=inputs, gen_kwargs=gen_kwargs, future=future)
        )
        return future.result(timeout=self._result_timeout_s)

    def _batching_loop(self) -> None:
        while True:
            try:
                try:
                    first_request = self._queue.get(
                        timeout=self._IDLE_LIVENESS_INTERVAL_S
                    )
                except queue.Empty:
                    # As long as anyone waits in `submit(...)`, the owning
                    # wrapper is referenced through the submitter's stack -
                    # an empty queue with a dead owner means the model got
                    # evicted and this thread can exit.
                    if self._runner_ref() is None:
                        LOGGER.debug(
                            "Dynamic batcher (%s): owning model released - "
                            "exiting batching loop",
                            self._name,
                        )
                        return None
                    continue
                self._process_batch(first_request=first_request)
            except Exception:
                # The batcher thread must never die while its model lives -
                # any error which is not handled by group execution is logged
                # and the loop continues with the next batch.
                LOGGER.exception(
                    "Dynamic batcher (%s): unexpected error in batching loop",
                    self._name,
                )

    def _process_batch(self, first_request: _PendingRequest) -> None:
        requests = [first_request]
        deadline = time.monotonic() + self._max_wait_s
        while len(requests) < self._max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                requests.append(self._queue.get(timeout=remaining))
            except queue.Empty:
                break
        now = time.monotonic()
        max_queue_wait_ms = max(
            (now - request.enqueued_at) * 1000 for request in requests
        )
        groups = self._group_by_generation_signature(requests=requests)
        self.stats["requests"] += len(requests)
        self.stats["max_batch_size_seen"] = max(
            self.stats["max_batch_size_seen"], len(requests)
        )
        LOGGER.debug(
            "Dynamic batcher (%s): collected %d request(s) in %d signature "
            "group(s); max queue wait: %.1fms",
            self._name,
            len(requests),
            len(groups),
            max_queue_wait_ms,
        )
        for group in groups:
            self._execute_group_safely(group=group)

    @staticmethod
    def _group_by_generation_signature(
        requests: List[_PendingRequest],
    ) -> List[List[_PendingRequest]]:
        groups: Dict[tuple, List[_PendingRequest]] = {}
        for request in requests:
            signature = _generation_signature(gen_kwargs=request.gen_kwargs)
            groups.setdefault(signature, []).append(request)
        return list(groups.values())

    def _execute_group_safely(self, group: List[_PendingRequest]) -> None:
        try:
            if len(group) == 1:
                self._execute_single(request=group[0])
            else:
                self._execute_batched(group=group)
        except Exception as error:
            # Last-resort safety net - no waiter may ever be left hanging.
            for request in group:
                if not request.future.done():
                    request.future.set_exception(error)

    def _execute_single(self, request: _PendingRequest) -> None:
        self.stats["batches"] += 1
        try:
            result = self._resolve_callable(reference=self._runner_ref)(
                request.inputs, request.gen_kwargs
            )
        except Exception as error:
            _handle_potential_cuda_oom(error=error)
            request.future.set_exception(error)
        else:
            request.future.set_result(result)

    def _execute_batched(self, group: List[_PendingRequest]) -> None:
        self.stats["batches"] += 1
        try:
            collated_inputs = self._resolve_callable(
                reference=self._collate_inputs_ref
            )([request.inputs for request in group])
            merged_gen_kwargs = _merge_gen_kwargs(group=group)
            output = self._resolve_callable(reference=self._runner_ref)(
                collated_inputs, merged_gen_kwargs
            )
            self.stats["batched_requests"] += len(group)
            self._dispatch_batched_output(group=group, output=output)
        except Exception as error:
            _handle_potential_cuda_oom(error=error)
            LOGGER.warning(
                "Dynamic batcher (%s): batched generation of %d request(s) "
                "failed (%s) - retrying each request serially",
                self._name,
                len(group),
                error,
            )
            self.stats["serial_fallbacks"] += 1
            self._run_serially(group=group)

    def _dispatch_batched_output(
        self, group: List[_PendingRequest], output: torch.Tensor
    ) -> None:
        expected_rows = sum(request.rows for request in group)
        if output.shape[0] != expected_rows:
            raise InputsCollationError(
                f"Batched generation returned {output.shape[0]} row(s), but "
                f"{expected_rows} row(s) were expected"
            )
        start = 0
        for request in group:
            request_output = output[start : start + request.rows]
            start += request.rows
            max_new_tokens = request.gen_kwargs.get("max_new_tokens")
            if isinstance(max_new_tokens, int):
                request_output = request_output[:, :max_new_tokens]
            request_output = strip_trailing_padding(
                generated_ids=request_output,
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_id,
            )
            request.future.set_result(request_output)

    def _run_serially(self, group: List[_PendingRequest]) -> None:
        for request in group:
            if request.future.done():
                continue
            self._execute_single(request=request)

    @staticmethod
    def _resolve_callable(reference: Callable[[], Optional[Callable]]) -> Callable:
        resolved = reference()
        if resolved is None:
            raise RuntimeError(
                "Model wrapper owning the dynamic batcher was released"
            )
        return resolved


def _make_callable_ref(target: Callable) -> Callable[[], Optional[Callable]]:
    # Bound methods are held weakly, so the batcher thread does not keep an
    # evicted model wrapper alive. Plain callables (functions, lambdas, test
    # doubles) are held strongly - a weak reference to them would die at once.
    if inspect.ismethod(target):
        return weakref.WeakMethod(target)
    return lambda: target


def _generation_signature(gen_kwargs: dict) -> tuple:
    # `max_new_tokens` is excluded from the signature - requests with
    # different limits can share a batch (the batch runs at the max limit and
    # each output is trimmed back). Everything else (do_sample, num_beams,
    # temperature, ...) changes generation semantics and must not be mixed.
    # A `None` limit cannot be merged with `max(...)`, so it stays part of
    # the signature and only batches with identical requests.
    return tuple(
        sorted(
            (key, repr(value))
            for key, value in gen_kwargs.items()
            if key != "max_new_tokens" or value is None
        )
    )


def _merge_gen_kwargs(group: List[_PendingRequest]) -> dict:
    merged = dict(group[0].gen_kwargs)
    limits = [request.gen_kwargs.get("max_new_tokens") for request in group]
    if all(isinstance(limit, int) for limit in limits):
        merged["max_new_tokens"] = max(limits)
    return merged


def _handle_potential_cuda_oom(error: Exception) -> None:
    if not isinstance(error, torch.cuda.OutOfMemoryError):
        return None
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def strip_trailing_padding(
    generated_ids: torch.Tensor,
    pad_token_id: Optional[int],
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Removes trailing all-padding columns from a single request's output.

    In a batched `generate()`, sequences which finish early get padded up to
    the longest sequence of the batch. A non-batched call only pads up to the
    longest sequence of the request, so trailing padding shared by all rows
    of the request is stripped to match the single-request output exactly.
    """
    if pad_token_id is None or generated_ids.numel() == 0:
        return generated_ids
    total_columns = generated_ids.shape[1]
    non_pad = generated_ids != pad_token_id
    column_numbers = torch.arange(
        1, total_columns + 1, device=generated_ids.device
    )
    keep_length = int((non_pad.long() * column_numbers).max().item())
    if (
        eos_token_id is not None
        and eos_token_id == pad_token_id
        and keep_length < total_columns
    ):
        # When PAD and EOS share the same token id, a finished sequence ends
        # with one terminating EOS - keep it, as non-batched generation would.
        keep_length += 1
    keep_length = max(keep_length, 1)
    return generated_ids[:, :keep_length]


def collate_left_padded_inputs(
    inputs_list: List[dict],
    pad_token_id: Optional[int],
    cat_keys: Sequence[str] = (),
    frame_pad_keys: Sequence[str] = (),
) -> dict:
    """
    Default collate for decoder-only VLMs: left-pads `input_ids` with
    `pad_token_id` (extending `attention_mask` with zeros accordingly) and
    concatenates per-request image tensors on dim 0.

    * `cat_keys` - tensors concatenated on dim 0 as-is (e.g. qwen-family
      `pixel_values` rows are flattened patches and `image_grid_thw` is
      `[num_images, 3]` - both batch by plain concatenation).
    * `frame_pad_keys` - `[batch, frames, ...]` tensors (e.g. smolvlm
      `pixel_values` / `pixel_attention_mask`) zero-padded on the frames
      dimension to the longest request before concatenation on dim 0.

    Raises `InputsCollationError` on any unsupported/inconsistent input - the
    batcher treats that as a failed batch and re-runs requests serially.
    """
    if not inputs_list:
        raise InputsCollationError("Cannot collate an empty list of inputs")
    key_set = set(inputs_list[0].keys())
    for inputs in inputs_list[1:]:
        if set(inputs.keys()) != key_set:
            raise InputsCollationError(
                "Cannot collate inputs with inconsistent keys: "
                f"{sorted(key_set)} vs {sorted(inputs.keys())}"
            )
    if "input_ids" not in key_set:
        raise InputsCollationError("Cannot collate inputs without `input_ids`")
    handled_keys = (
        {"input_ids", "attention_mask"} | set(cat_keys) | set(frame_pad_keys)
    )
    unsupported_keys = key_set - handled_keys
    if unsupported_keys:
        raise InputsCollationError(
            f"Cannot collate inputs with unsupported keys: {sorted(unsupported_keys)}"
        )
    max_length = max(inputs["input_ids"].shape[-1] for inputs in inputs_list)
    requires_padding = any(
        inputs["input_ids"].shape[-1] != max_length for inputs in inputs_list
    )
    if requires_padding and pad_token_id is None:
        raise InputsCollationError(
            "Cannot left-pad inputs without a pad token id"
        )
    if requires_padding and "attention_mask" not in key_set:
        raise InputsCollationError(
            "Cannot left-pad inputs without `attention_mask`"
        )
    input_ids_rows, attention_mask_rows = [], []
    for inputs in inputs_list:
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        missing_columns = max_length - input_ids.shape[-1]
        if missing_columns > 0:
            pad_block = torch.full(
                (input_ids.shape[0], missing_columns),
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([pad_block, input_ids], dim=1)
            attention_mask = torch.cat(
                [
                    torch.zeros(
                        (attention_mask.shape[0], missing_columns),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )
        input_ids_rows.append(input_ids)
        if attention_mask is not None:
            attention_mask_rows.append(attention_mask)
    collated = {"input_ids": torch.cat(input_ids_rows, dim=0)}
    if "attention_mask" in key_set:
        collated["attention_mask"] = torch.cat(attention_mask_rows, dim=0)
    for key in cat_keys:
        if key in key_set:
            collated[key] = torch.cat([inputs[key] for inputs in inputs_list], dim=0)
    for key in frame_pad_keys:
        if key in key_set:
            collated[key] = _cat_with_frame_padding(
                tensors=[inputs[key] for inputs in inputs_list], key=key
            )
    return collated


def _cat_with_frame_padding(tensors: List[torch.Tensor], key: str) -> torch.Tensor:
    trailing_shapes = {tuple(tensor.shape[2:]) for tensor in tensors}
    if len(trailing_shapes) > 1:
        raise InputsCollationError(
            f"Cannot collate `{key}` - incompatible shapes: {sorted(trailing_shapes)}"
        )
    max_frames = max(tensor.shape[1] for tensor in tensors)
    padded_tensors = []
    for tensor in tensors:
        missing_frames = max_frames - tensor.shape[1]
        if missing_frames > 0:
            pad_block = tensor.new_zeros(
                (tensor.shape[0], missing_frames) + tuple(tensor.shape[2:])
            )
            tensor = torch.cat([tensor, pad_block], dim=1)
        padded_tensors.append(tensor)
    return torch.cat(padded_tensors, dim=0)


_BATCHER_CREATION_LOCK = Lock()

DEFAULT_BATCH_CAT_KEYS = ("pixel_values", "image_grid_thw", "pixel_attention_mask")


class DynamicBatchingMixin:
    """
    Mixin wiring an HF VLM wrapper into a `DynamicBatcher`.

    The hosting wrapper must define `_run_locked_generation(inputs, gen_kwargs)`
    (the existing lock-protected `generate()` body, returning the trimmed
    new-token tensor) and expose `self._processor` with a `tokenizer`.

    One `DynamicBatcher` is created lazily per model instance, only when
    dynamic batching is enabled and the first request arrives. The runner
    acquires the wrapper's existing `self._lock` - uncontended when batching
    is on (all requests funnel through one batcher thread), but still correct
    if anything else calls the locked generation directly.

    Image tensor collation is overridable per model family with the
    `BATCH_CAT_KEYS` / `BATCH_FRAME_PAD_KEYS` class attributes or a custom
    `_collate_inputs` method.
    """

    BATCH_CAT_KEYS: Tuple[str, ...] = DEFAULT_BATCH_CAT_KEYS
    BATCH_FRAME_PAD_KEYS: Tuple[str, ...] = ()

    def _dynamic_batching_enabled(self) -> bool:
        return configuration.INFERENCE_MODELS_DYNAMIC_BATCHING_ENABLED

    def _submit_to_dynamic_batcher(
        self, inputs: dict, gen_kwargs: dict
    ) -> torch.Tensor:
        return self._get_dynamic_batcher().submit(
            inputs=inputs, gen_kwargs=gen_kwargs
        )

    def _get_dynamic_batcher(self) -> DynamicBatcher:
        batcher = getattr(self, "_dynamic_batcher", None)
        if batcher is not None:
            return batcher
        with _BATCHER_CREATION_LOCK:
            batcher = getattr(self, "_dynamic_batcher", None)
            if batcher is None:
                batcher = DynamicBatcher(
                    runner=self._run_locked_generation,
                    collate_inputs=self._collate_inputs,
                    max_batch_size=configuration.INFERENCE_MODELS_DYNAMIC_BATCH_MAX_SIZE,
                    max_wait_ms=configuration.INFERENCE_MODELS_DYNAMIC_BATCH_MAX_WAIT_MS,
                    result_timeout_s=configuration.INFERENCE_MODELS_DYNAMIC_BATCH_RESULT_TIMEOUT_S,
                    pad_token_id=self._generation_pad_token_id(),
                    eos_token_id=self._generation_eos_token_id(),
                    name=type(self).__name__,
                )
                self._dynamic_batcher = batcher
        return batcher

    def _run_locked_generation(self, inputs: dict, gen_kwargs: dict) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement `_run_locked_generation(...)` "
            "to use dynamic batching"
        )

    def _collate_inputs(self, inputs_list: List[dict]) -> dict:
        return collate_left_padded_inputs(
            inputs_list=inputs_list,
            pad_token_id=self._generation_pad_token_id(),
            cat_keys=self.BATCH_CAT_KEYS,
            frame_pad_keys=self.BATCH_FRAME_PAD_KEYS,
        )

    def _generation_pad_token_id(self) -> Optional[int]:
        return self._processor.tokenizer.pad_token_id

    def _generation_eos_token_id(self) -> Optional[int]:
        return self._processor.tokenizer.eos_token_id
