"""``v2/identify_outliers@v1``: flag embeddings unlike this source's recent ones.

A port of V1 ``roboflow_core/identify_outliers@v1`` (tensor variant). One call
per embedding, on the embedding's device::

    e = embedding / |embedding|               (a zero vector stays zero)
    window <- e                               (trimmed to the last window_size)
    warming_up = samples < warmup             (counted with this call)
    mu = sum(window) / |sum(window)|          von Mises-Fisher mean direction;
                                              e itself when the sum is zero
    scores = window @ mu                      e's own score is scores[-1]
    percentile = (#scores < e's + #scores <= e's) / (2 n)      mid-CDF rank
    is_outlier = percentile < t or percentile > 1 - t

Warmup returns ``(False, 0.5, True)`` and still fills the window.

V1 defects corrected here, which change V1's numbers on purpose:

* Ties: V1 ranked from the left, so a constant stream got percentile 0 and
  was flagged on every call. Tied scores now share the middle of their
  ranks, so a constant window or a single embedding gives 0.5.
* V1 ranked a second dot product ``mu . e`` against the matrix row of ``e``;
  their float rounding differed and could move e by one rank. The rank now
  uses e's own row of ``scores``.
* A smaller selected ``window_size`` trims the window in the same call; V1
  dropped at most one embedding per call.
* A NaN or Inf embedding fails with ``ValueError`` and changes nothing.

Without ties, an untied extreme gets percentile ``1 / (2n)`` (or
``1 - 1 / (2n)``), not V1's 0 (or ``(n - 1) / n``). A window of ``n``
embeddings can fire the strict tail ``t`` only when ``1 / (2n) < t``: with
the default t=0.05 that needs n >= 11, so a window of 8 cannot fire at all.
The default window of 32 can.

State follows ``_state``, as in ``v2/identify_changes@v1``: ``scope="source"``
(default) keeps one window per source, ``"global"`` one per block as V1 did;
``max_sources`` bounds the windows and fails with ``SamplingCapacityError``.
Each call reads one scalar from the device for the finite check, and after
warmup one more for the rank.
"""

from typing import Any, Dict, List

import torch
from pydantic import Field, StrictInt
from roboflow_workflows.v2.blocks.sampling._state import (
    DEFAULT_MAX_SOURCES,
    Scope,
    SourceStates,
    UnitFloat,
    WindowSize,
    check_fraction,
    check_like,
    check_window_size,
    state_key,
    unit_vector,
)
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.data.predictions import EMBEDDING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

__all__ = ["IdentifyOutliersBlock"]


class _Window:
    """Unit embeddings of one source, oldest first, and the samples seen."""

    def __init__(self) -> None:
        self.samples = 0
        self.embeddings: List[torch.Tensor] = []


class IdentifyOutliersBlock(Block):
    """Flag an embedding whose direction ranks at either end of its source's window."""

    type = "v2/identify_outliers@v1"
    metadata = {
        "name": "Identify Outliers",
        "port_of": "roboflow_core/identify_outliers@v1",
        "section": "video",
    }
    outputs = {
        "is_outlier": Output(
            BOOLEAN_KIND,
            source="embedding",
            description="Whether the percentile is below t or above 1 - t.",
        ),
        "percentile": Output(
            FLOAT_KIND,
            source="embedding",
            description="Mid-CDF rank of the alignment with the mean direction, 0 to 1.",
        ),
        "warming_up": Output(
            BOOLEAN_KIND,
            source="embedding",
            description="Whether fewer than warmup embeddings were seen.",
        ),
    }

    class Params(BlockParams):
        embedding: Ref(EMBEDDING_KIND) = Field(
            description="Embedding of the current sample, e.g. from CLIP.",
            examples=["$steps.clip.embedding"],
        )
        threshold_percentile: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.05,
            description=(
                "From 0 to 1: percentiles below t or above 1 - t are outliers."
            ),
            examples=[0.05, "$inputs.threshold_percentile"],
        )
        warmup: StrictInt | Ref(INTEGER_KIND) = Field(
            default=3,
            description=(
                "Sample number per source at which outlier scoring starts, "
                "including the current sample. With 3, the first two warm up."
            ),
            examples=[3, 10],
        )
        window_size: WindowSize | Ref(INTEGER_KIND) = Field(
            default=32,
            description=(
                "Recent embeddings kept per source, at least 1. V1's "
                "'unlimited' null never worked and is not accepted."
            ),
            examples=[32, 100],
        )
        scope: Scope = Field(
            default="source",
            description=(
                "'source' keeps a window per source; 'global' shares one "
                "window across sources, as V1 did."
            ),
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources with a window. A call of one more source fails "
                "with SamplingCapacityError; no window is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._states: SourceStates[_Window] = SourceStates(self.type, create=_Window)

    def run(
        self,
        *,
        embedding: torch.Tensor,
        threshold_percentile: float,
        warmup: int,
        window_size: int,
        scope: str,
        max_sources: int,
    ) -> Dict[str, Any]:
        """Add this embedding to its source's window and rank it there.

        Args:
            embedding: 1-D floating embedding of the current sample.
            threshold_percentile: Two-sided outlier threshold, 0 to 1.
            warmup: Embeddings, this one included, before outliers can be
                flagged.
            window_size: Window length, at least 1.
            scope: ``"source"`` or ``"global"``.
            max_sources: Most kept source windows.

        Returns:
            ``is_outlier``, ``percentile`` and ``warming_up``.

        Raises:
            ValueError: On an invalid embedding or parameter, or an embedding
                unlike its source's.
            SamplingCapacityError: When a new source would exceed
                ``max_sources``.
        """
        threshold = check_fraction(threshold_percentile, name="threshold_percentile")
        window_size = check_window_size(window_size)
        e = unit_vector(embedding)
        key = state_key(self, scope=scope)

        with self._states.lock:
            window = self._states.get(key)
            if window is not None and window.embeddings:
                check_like(e, kept=window.embeddings[0], block_type=self.type)
            if window is None:
                window = self._states.add(key, max_sources=max_sources)
            window.samples += 1
            window.embeddings.append(e)
            del window.embeddings[:-window_size]
            if window.samples < warmup:
                return {"is_outlier": False, "percentile": 0.5, "warming_up": True}

            kept = torch.stack(window.embeddings)

        percentile = _percentile(kept)
        is_outlier = percentile < threshold or percentile > 1 - threshold
        result = {
            "is_outlier": is_outlier,
            "percentile": percentile,
            "warming_up": False,
        }

        return result

    def reset_state(self) -> None:
        """Forget every source's window."""
        with self._states.lock:
            self._states.clear()


def _percentile(kept: torch.Tensor) -> float:
    """Mid-CDF rank of the newest embedding's score among the window's, 0 to 1.

    ``kept[-1]`` is the current embedding. Its score is its own entry of
    ``kept @ mu``, so it is compared with exactly the rounding of the others.
    """
    scores = kept @ _mean_direction(kept)
    score = scores[-1]
    twice_rank = int(((scores < score).sum() + (scores <= score).sum()).item())
    percentile = twice_rank / (2 * len(scores))

    return percentile


def _mean_direction(kept: torch.Tensor) -> torch.Tensor:
    """vMF mean direction of the window; the newest embedding when the sum is zero.

    A single embedding is its own mean direction. V1 also estimated the
    concentration kappa but never used it.
    """
    total = kept.sum(dim=0)
    length = torch.linalg.norm(total)
    # Chosen on the device, so the fit check needs no host sync.
    mu = torch.where(
        length == 0,
        kept[-1],
        total / torch.where(length == 0, torch.ones_like(length), length),
    )

    return mu
