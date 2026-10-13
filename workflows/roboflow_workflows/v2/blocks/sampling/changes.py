"""``v2/identify_changes@v1``: flag an unusual rate of embedding change, per source.

A port of V1 ``roboflow_core/identify_changes@v1`` (tensor variant). One call
per embedding::

    e  = embedding / |embedding|                  (a zero vector stays zero)
    cs = cosine(e, average) in [-1, 1]            (from the second sample on;
                                                   0 when either is zero)
    cs statistics  <- strategy(cs)                (EMA / SMA / Sliding Window)
    cs_std <= noise floor:  z = 0, percentile = 0.5
    otherwise:              z = (cs - cs_avg) / cs_std, percentile = 1 - Phi(z)
    is_outlier = not warming_up and (p <= t/2 or p >= 1 - t/2)
    average, std   <- strategy(e)                 per dimension, on e's device

V1's arithmetic and order are kept: ``percentile`` and ``z_score`` are
computed during warmup too, the first embedding never enters the sliding
window, and the SMA of ``cs`` divides by the embedding count.

V1 defects corrected here, which change V1's numbers on purpose:

* Constant embedding: V1 turned float rounding of the cosine into large
  z-scores and flagged it. A cosine std at most ``2 * eps`` of the
  embedding's dtype is rounding noise, so z stays 0 (see ``_noise_floor``).
  The cosine is also clamped to [-1, 1].
* Zero vector: its cosine is 0, not V1's 0/0 = NaN that kept the statistics
  NaN for every later call.
* A smaller selected ``window_size`` trims both windows to that length in
  the same call; V1 dropped at most one value per call.
* A NaN or Inf embedding fails with ``ValueError`` and changes nothing.

State follows ``_state``: ``scope="source"`` (default) keeps one state per
source, ``"global"`` one per block as V1 did; ``max_sources`` bounds the
states and fails with ``SamplingCapacityError``, never evicting. A processing
reset or ``reset_state()`` drops every state. The engine orders the calls of
one domain; one lock per block keeps each update whole across domains. An
embedding of another length, dtype or device than its source's state fails
with ``ValueError`` and changes nothing.

Device: the embedding statistics stay in torch on the embedding's device and
dtype. The cosine-similarity statistics are host floats, as V1's numpy
scalars were. Each call reads two scalars from the device: the finite check
and, from the second call on, the cosine.
"""

import math
from typing import Any, Dict, List, Literal, Optional

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

__all__ = ["EMA", "SLIDING_WINDOW", "SMA", "IdentifyChangesBlock"]

EMA = "Exponential Moving Average (EMA)"
SMA = "Simple Moving Average (SMA)"
SLIDING_WINDOW = "Sliding Window"

Strategy = Literal[
    "Exponential Moving Average (EMA)",
    "Simple Moving Average (SMA)",
    "Sliding Window",
]


def _sqrt(value: float) -> float:
    # A variance below 0 is a rounding residue of 0.
    root = math.sqrt(max(value, 0.0))

    return root


def _cosine(e: torch.Tensor, average: torch.Tensor) -> float:
    """V1's cosine, clamped to [-1, 1]; 0 when either vector is zero."""
    norms = torch.sqrt(torch.dot(e, e) * torch.dot(average, average))
    # A zero vector has no direction; its dot product is 0, so 0 / 1 = 0.
    cosine = torch.dot(e, average) / torch.where(
        norms == 0, torch.ones_like(norms), norms
    )
    cs = float(cosine.clamp(-1, 1).item())

    return cs


def _noise_floor(dtype: torch.dtype) -> float:
    """Noise floor for cosine std in the embedding's dtype.

    The cosine is computed in the embedding's dtype. For a constant
    embedding its std stayed at most 0.71 eps in float32 and float64 sweeps
    (dimensions 3 to 8192, every strategy, clamped). Twice eps leaves margin
    over those measurements. Small real embedding changes can also fall below
    this cosine-resolution floor and count as no change.
    """
    floor = 2 * torch.finfo(dtype).eps

    return floor


class _ChangeState:
    """Running statistics of one source; names follow V1's attributes."""

    def __init__(self) -> None:
        self.samples = 0
        self.average: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.var: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None
        self.window: List[torch.Tensor] = []
        self.cs_avg: Optional[float] = None
        self.cs_std = 0.0
        self.cs_var = 0.0
        self.cs_m2 = 0.0
        self.cs_window: List[float] = []

    def observe_similarity(
        self, cs: float, *, strategy: str, alpha: float, window_size: int
    ) -> None:
        if self.cs_avg is None:
            self.cs_avg = cs
            self.cs_std = self.cs_var = self.cs_m2 = 0.0
        elif strategy == EMA:
            self.cs_avg = (1 - alpha) * self.cs_avg + alpha * cs
            diff = cs - self.cs_avg
            self.cs_var = (1 - alpha) * self.cs_var + alpha * (diff**2)
            self.cs_std = _sqrt(self.cs_var)
        elif strategy == SMA:
            count = self.samples + 1
            delta = cs - self.cs_avg
            self.cs_avg = cs / count + self.cs_avg * self.samples / count
            delta2 = cs - self.cs_avg
            self.cs_m2 = self.cs_m2 + delta * delta2
            self.cs_std = _sqrt(self.cs_m2 / (count - 1))
        else:
            self.cs_window.append(cs)
            del self.cs_window[:-window_size]
            count = len(self.cs_window)
            self.cs_avg = sum(self.cs_window) / count
            squares = sum((value - self.cs_avg) ** 2 for value in self.cs_window)
            self.cs_std = _sqrt(squares / count)

    def observe_embedding(
        self, e: torch.Tensor, *, strategy: str, alpha: float, window_size: int
    ) -> None:
        if self.average is None:
            self.average = e
            self.std = torch.zeros_like(e)
            self.var = torch.zeros_like(e)
            self.m2 = torch.zeros_like(e)
        elif strategy == EMA:
            self.average = (1 - alpha) * self.average + alpha * e
            diff = e - self.average
            self.var = (1 - alpha) * self.var + alpha * (diff**2)
            self.std = torch.sqrt(self.var)
        elif strategy == SMA:
            # Welford's method.
            count = self.samples + 1
            delta = e - self.average
            self.average = self.average + delta / count
            delta2 = e - self.average
            self.m2 = self.m2 + delta * delta2
            self.std = torch.sqrt(self.m2 / (count - 1))
        else:
            self.window.append(e)
            del self.window[:-window_size]
            window = torch.stack(self.window)
            self.average = window.mean(dim=0)
            self.std = window.std(dim=0, unbiased=False)


class IdentifyChangesBlock(Block):
    """Flag embeddings that change faster or slower than this source usually does.

    ``average`` and ``std`` are the source's running statistics after this
    call, per dimension, on the embedding's device. They are the kept state
    objects: do not change them in place (declared in ``retains_outputs``).
    """

    type = "v2/identify_changes@v1"
    metadata = {
        "name": "Identify Changes",
        "port_of": "roboflow_core/identify_changes@v1",
        "section": "video",
    }
    outputs = {
        "is_outlier": Output(
            BOOLEAN_KIND,
            source="embedding",
            description="Whether the rate of change is unusual; never during warmup.",
        ),
        "percentile": Output(
            FLOAT_KIND,
            source="embedding",
            description="1 - Phi(z_score), from 0 to 1; 0.5 without statistics.",
        ),
        "z_score": Output(
            FLOAT_KIND,
            source="embedding",
            description="Cosine similarity to the average, in standard deviations.",
        ),
        "average": Output(
            EMBEDDING_KIND,
            source="embedding",
            description="Running per-dimension average of unit embeddings.",
        ),
        "std": Output(
            EMBEDDING_KIND,
            source="embedding",
            description="Running per-dimension standard deviation.",
        ),
        "warming_up": Output(
            BOOLEAN_KIND,
            source="embedding",
            description="Whether fewer than warmup earlier embeddings were seen.",
        ),
    }
    retains_outputs = ("average", "std")

    class Params(BlockParams):
        embedding: Ref(EMBEDDING_KIND) = Field(
            description="Embedding of the current sample, e.g. from CLIP.",
            examples=["$steps.clip.embedding"],
        )
        strategy: Strategy = Field(
            default=EMA,
            description=(
                "How running statistics are kept: EMA weights recent samples "
                "by smoothing_factor, SMA weights all equally, Sliding Window "
                "uses the last window_size."
            ),
        )
        threshold_percentile: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.2,
            description=(
                "From 0 to 1: percentiles at most t/2 or at least 1 - t/2 are "
                "outliers."
            ),
            examples=[0.2, "$inputs.threshold_percentile"],
        )
        warmup: StrictInt | Ref(INTEGER_KIND) = Field(
            default=3,
            description="Embeddings per source before outliers can be flagged.",
            examples=[3, 10],
        )
        smoothing_factor: Optional[UnitFloat] | Ref(FLOAT_KIND) = Field(
            default=0.1,
            description="EMA weight of the newest sample, 0 to 1; EMA only.",
            examples=[0.1, 0.25],
        )
        window_size: Optional[WindowSize] | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Samples in the window, at least 1; Sliding Window only.",
            examples=[10, 50],
        )
        scope: Scope = Field(
            default="source",
            description=(
                "'source' keeps statistics per source; 'global' shares one "
                "state across sources, as V1 did."
            ),
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources with statistics. A call of one more source fails "
                "with SamplingCapacityError; no state is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

    def __init__(self) -> None:
        self._states: SourceStates[_ChangeState] = SourceStates(
            self.type, create=_ChangeState
        )

    def run(
        self,
        *,
        embedding: torch.Tensor,
        strategy: str,
        threshold_percentile: float,
        warmup: int,
        smoothing_factor: Optional[float],
        window_size: Optional[int],
        scope: str,
        max_sources: int,
    ) -> Dict[str, Any]:
        """Score this embedding against its source's statistics, then update them.

        Args:
            embedding: 1-D floating embedding of the current sample.
            strategy: One of the three V1 strategy names.
            threshold_percentile: Two-sided outlier threshold, 0 to 1.
            warmup: Embeddings before outliers can be flagged.
            smoothing_factor: EMA weight, 0 to 1; required for EMA.
            window_size: Window length, at least 1; required for Sliding Window.
            scope: ``"source"`` or ``"global"``.
            max_sources: Most kept source states.

        Returns:
            ``is_outlier``, ``percentile``, ``z_score``, ``average``, ``std``
            and ``warming_up``.

        Raises:
            ValueError: On an invalid embedding or parameter, a missing
                strategy parameter, or an embedding unlike its source's.
            SamplingCapacityError: When a new source would exceed
                ``max_sources``.
        """
        threshold = check_fraction(threshold_percentile, name="threshold_percentile")
        alpha = smoothing_factor
        if strategy == EMA:
            if smoothing_factor is None:
                raise ValueError(
                    f"{self.type}: strategy {EMA!r} needs smoothing_factor"
                )
            alpha = check_fraction(smoothing_factor, name="smoothing_factor")
        if strategy == SLIDING_WINDOW:
            if window_size is None:
                raise ValueError(
                    f"{self.type}: strategy {SLIDING_WINDOW!r} needs window_size"
                )
            window_size = check_window_size(window_size)
        e = unit_vector(embedding)
        key = state_key(self, scope=scope)

        with self._states.lock:
            state = self._states.get(key)
            if state is not None and state.average is not None:
                check_like(e, kept=state.average, block_type=self.type)
            if state is None:
                state = self._states.add(key, max_sources=max_sources)
            result = self._update(
                state,
                e,
                strategy=strategy,
                threshold=threshold,
                warmup=warmup,
                alpha=alpha,
                window_size=window_size,
            )

        return result

    def reset_state(self) -> None:
        """Forget every source's statistics."""
        with self._states.lock:
            self._states.clear()

    @staticmethod
    def _update(
        state: _ChangeState,
        e: torch.Tensor,
        *,
        strategy: str,
        threshold: float,
        warmup: int,
        alpha: Optional[float],
        window_size: Optional[int],
    ) -> Dict[str, Any]:
        z_score, percentile = 0.0, 0.5
        if state.average is not None:
            cs = _cosine(e, state.average)
            state.observe_similarity(
                cs, strategy=strategy, alpha=alpha, window_size=window_size
            )
            # No variance beyond rounding (e.g. a static scene): nothing to
            # compare against.
            if state.cs_std > _noise_floor(e.dtype):
                z_score = (cs - state.cs_avg) / state.cs_std
                percentile = 1 - 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

        warming_up = state.samples < warmup
        is_outlier = not warming_up and (
            percentile <= threshold / 2 or percentile >= 1 - threshold / 2
        )
        state.observe_embedding(
            e, strategy=strategy, alpha=alpha, window_size=window_size
        )
        state.samples += 1
        result = {
            "is_outlier": is_outlier,
            "percentile": float(percentile),
            "z_score": float(z_score),
            "average": state.average,
            "std": state.std,
            "warming_up": warming_up,
        }

        return result
