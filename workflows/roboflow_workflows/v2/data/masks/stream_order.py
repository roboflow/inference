"""CUDA stream ordering for mask results shared between callers.

A mask result may be computed on one CUDA stream and read on others. The
helpers here order every reader after the producer without a host
synchronization::

    producer stream:  compute ─► StreamReadiness.record   (event E)
    reader stream:    order_current_stream(E, tensors)    waits for E, if it is
                                                          another stream, then
                                                          record_stream(tensors)

``StreamOrderedOnce`` combines both for a result computed at most once: the
first caller computes and publishes it with its event, and every caller,
including the first, gets it ordered for its own current stream. CPU results
skip all CUDA work.
"""

import threading
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from roboflow_workflows.v2.data.row_identity import checked_row_record

from inference_models.models.base.instance_segmentation import InstanceDetections


@dataclass(frozen=True)
class StreamReadiness:
    """A CUDA event recorded on the stream that produced some tensors."""

    stream: Any
    event: Any

    @classmethod
    def record(cls, device: torch.device) -> Optional["StreamReadiness"]:
        """Record an event on the current stream of ``device``; ``None`` on CPU."""
        if device.type != "cuda":
            return None

        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)

        return cls(stream=stream, event=event)


def order_current_stream(
    readiness: Optional[StreamReadiness], tensors: Sequence[torch.Tensor]
) -> None:
    """Order the caller's current stream after ``readiness``.

    No host synchronization. Two separate guarantees:

    - Readiness: on a stream other than the one that recorded the event, the
      stream waits for the event before it reads the tensors.
    - Allocation lifetime: the caching allocator learns that the current
      stream uses the tensors, even on the readiness stream, so their memory
      is not reused before that stream's work completes. A borrowed tensor
      may have been allocated on yet another stream.
    """
    if readiness is None:
        return

    stream = torch.cuda.current_stream(readiness.stream.device)
    if stream != readiness.stream:
        stream.wait_event(readiness.event)
    for tensor in tensors:
        tensor.record_stream(stream)


class StreamOrderedOnce:
    """Dense predictions computed at most once and shared across streams.

    Concurrent first callers wait for one computation. A failure publishes
    nothing, and the next call retries. The published result is shared by
    every caller and read-only by contract.

    Args:
        lock: Held during the computation. Pass the owner's lock when another
            owner path must not run at the same time as the computation.
    """

    def __init__(self, lock: Optional[threading.Lock] = None) -> None:
        self._lock = lock if lock is not None else threading.Lock()
        self._published: Optional[
            Tuple[InstanceDetections, Optional[StreamReadiness], List[torch.Tensor]]
        ] = None

    @property
    def is_set(self) -> bool:
        """``True`` once a computation has succeeded."""
        return self._published is not None

    def get(self, compute: Callable[[], InstanceDetections]) -> InstanceDetections:
        """The published result, computed by ``compute`` on the first call.

        On CUDA, ``compute`` runs on the first caller's current stream; its
        event is recorded there right after it returns. Every caller gets the
        result ordered for its own current stream.

        Args:
            compute: Builds the result on the current stream. Called at most
                once per successful publication, under the lock.

        Returns:
            The shared result.

        Raises:
            ContractError: When the result holds a malformed or stale row
                record; nothing is published.
        """
        published = self._published
        if published is None:
            with self._lock:
                if self._published is None:
                    predictions = compute()
                    tensors = _prediction_tensors(predictions)
                    ready = StreamReadiness.record(predictions.mask.device)
                    self._published = (predictions, ready, tensors)
                published = self._published

        predictions, ready, tensors = published
        order_current_stream(ready, tensors)

        return predictions


def _prediction_tensors(predictions: InstanceDetections) -> List[torch.Tensor]:
    """Row tensors of the predictions, including row-identity ids when present.

    Raises:
        ContractError: When the predictions hold a malformed or stale row record.
    """
    tensors = [
        predictions.mask,
        predictions.xyxy,
        predictions.class_id,
        predictions.confidence,
    ]
    record = checked_row_record(
        predictions.image_metadata,
        rows=int(predictions.xyxy.shape[0]),
        device=predictions.xyxy.device,
        name="predictions.image_metadata",
    )
    if record is not None and record.ids is not None:
        tensors.append(record.ids)

    return tensors
