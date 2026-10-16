"""Empirical arithmetic qualification against the installed torchvision binary.

A unique sampled match is necessary, not proof of equality on every possible
input. Calibration runs once per stage/device; CUDA and compilation errors
propagate. No guessed arithmetic is used when qualification is inconclusive.
"""

from typing import Any, Dict, Sequence, Tuple

import torch
import torchvision

from .nms_kernels import ARITHMETIC, TritonNmsKernels


class CalibrationError(RuntimeError):
    """The installed oracle did not uniquely qualify an arithmetic variant.

    Args:
        message: Human-readable qualification failure.
        report: The ``calibrate`` report that failed to qualify.
    """

    def __init__(self, message: str, *, report: Dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


CALIBRATION_THRESHOLDS = (0.3, 0.45, 0.5, 0.7)
# Pair geometry per calibration round: (name, origin, origin span, min side, max side).
CALIBRATION_SCALES = (
    ("small", 0.0, 64.0, 1.0, 64.0),
    ("class-offset", 1.0e4, 9.0e4, 10.0, 600.0),
)


def build_kernels(device: Any) -> Tuple[TritonNmsKernels, Dict[str, Any]]:
    """The Triton kernels for ``device``, with the calibrated IoU arithmetic.

    Args:
        device: A CUDA device.

    Returns:
        The kernels and the calibration report.

    Raises:
        RuntimeError: Not a CUDA device.
        CalibrationError: Calibration did not find exactly one variant equal
            to the oracle on every checked pair.
    """
    if torch.device(device).type != "cuda":
        raise RuntimeError(f"Triton NMS kernels need a CUDA device, got {device}")

    report = calibrate(device)
    matching = report["matching"]
    if len(matching) != 1:
        raise CalibrationError(
            f"IoU calibration needs exactly one variant equal to "
            f"torchvision.ops.nms on every checked pair, got {matching}; "
            f"mismatches per variant: {report['mismatches']}",
            report=report,
        )

    kernels = TritonNmsKernels(arithmetic=matching[0])

    return kernels, report


def calibrate(
    device: Any,
    *,
    candidate_pairs: int = 32768,
    checked_disagreeing: int = 256,
    checked_agreeing: int = 32,
    thresholds: Sequence[float] = CALIBRATION_THRESHOLDS,
    seed: int = 0,
) -> Dict[str, Any]:
    """Which ``ARITHMETIC`` variants agree with ``torchvision.ops.nms`` on samples.

    Per threshold and scale: build ``candidate_pairs`` isolated 2-box images
    whose float64 IoU equals the threshold before the float32 cast, so their
    float32 IoU lies a few ulp around it. Run every variant over all of them in
    one call each, then ask the oracle (one ``nms`` call per pair, its own
    host wait) about pairs where variants disagree plus a few where they agree.

    Args:
        device: CUDA device.
        candidate_pairs: Pairs generated per round.
        checked_disagreeing: Disagreeing pairs checked per round, at most.
        checked_agreeing: Agreeing pairs checked per round (sanity).
        thresholds: IoU thresholds, one round per threshold and scale.
        seed: Generator seed.

    Returns:
        JSON-friendly report: rounds, mismatches per variant, ``matching``
        (variants with no mismatch on the checked pairs) and the torchvision
        version.
    """
    mismatches = dict.fromkeys(ARITHMETIC, 0)
    rounds = []
    for round_index, (threshold, scale) in enumerate(
        (threshold, scale) for threshold in thresholds for scale in CALIBRATION_SCALES
    ):
        pairs = _boundary_pairs(
            candidate_pairs, threshold=threshold, scale=scale, seed=seed + round_index
        ).to(device)
        decisions = torch.stack(
            [
                _second_box_kept(pairs, threshold=threshold, arithmetic=name)
                for name in ARITHMETIC
            ]
        )
        disagree = (decisions != decisions[0]).any(0)
        checked = torch.cat(
            (
                disagree.nonzero().flatten()[:checked_disagreeing],
                (~disagree).nonzero().flatten()[:checked_agreeing],
            )
        ).tolist()
        expected = torch.tensor(
            [_oracle_second_box_kept(pairs[index], threshold) for index in checked],
            dtype=torch.bool,
            device=pairs.device,
        )
        wrong = (decisions[:, checked] != expected).sum(1).tolist()
        for name, count in zip(ARITHMETIC, wrong):
            mismatches[name] += count
        rounds.append(
            {
                "threshold": threshold,
                "scale": scale[0],
                "pairs": candidate_pairs,
                "disagreeing": int(disagree.sum()),
                "checked": len(checked),
                "mismatches": dict(zip(ARITHMETIC, wrong)),
            }
        )

    report = {
        "calibrated": True,
        "oracle": f"torchvision.ops.nms {torchvision.__version__}, one call per pair",
        "rounds": rounds,
        "mismatches": mismatches,
        "matching": [name for name, count in mismatches.items() if count == 0],
    }

    return report


def _second_box_kept(
    pairs: torch.Tensor, *, threshold: float, arithmetic: str
) -> torch.Tensor:
    # Every pair is its own image of two boxes; box 0 has the higher score.
    count = pairs.shape[0]
    device = pairs.device
    counts = torch.full((count,), 2, dtype=torch.long, device=device)
    rows = torch.zeros((count, 2), dtype=torch.long, device=device)
    kept = torch.zeros((count,), dtype=torch.long, device=device)
    TritonNmsKernels(arithmetic=arithmetic).keep(
        pairs.reshape(-1, 4).contiguous(),
        torch.tensor([0, 1], device=device).repeat(count),
        starts=torch.arange(count, device=device) * 2,
        counts=counts,
        max_count=2,
        iou_threshold=threshold,
        rows=rows,
        kept=kept,
    )
    second_kept = kept == 2

    return second_kept


def _oracle_second_box_kept(pair: torch.Tensor, threshold: float) -> bool:
    scores = torch.tensor([0.9, 0.8], device=pair.device)
    keep = torchvision.ops.nms(pair, scores, threshold)
    second_kept = keep.numel() == 2

    return second_kept


def _boundary_pairs(
    count: int, *, threshold: float, scale: Tuple[Any, ...], seed: int
) -> torch.Tensor:
    """``count`` box pairs whose exact IoU is ``threshold`` before the float32 cast.

    Box B is box A moved by (dx, dy) with the same size; for a random dy,
    dx solves (w - dx)(h - dy) / (2wh - (w - dx)(h - dy)) = threshold.

    Args:
        count: Number of pairs.
        threshold: Target IoU, in (0, 1).
        scale: (name, origin, origin span, min side, max side), e.g. one
            ``CALIBRATION_SCALES`` entry.
        seed: Generator seed.

    Returns:
        (count, 2, 4) float32 xyxy boxes on the CPU; row 0 is the higher score.
    """
    _, origin, span, min_side, max_side = scale
    generator = torch.Generator().manual_seed(seed)

    def uniform(low: float, high: float) -> torch.Tensor:
        values = torch.rand(count, generator=generator, dtype=torch.float64)
        return low + (high - low) * values

    x, y = uniform(origin, origin + span), uniform(origin, origin + span)
    width, height = uniform(min_side, max_side), uniform(min_side, max_side)
    overlap = 2 * threshold / (1 + threshold)  # intersection / (w h)
    dy = uniform(0.0, 1.0) * height * (1 - overlap)
    dx = width - overlap * width * height / (height - dy)
    first = torch.stack((x, y, x + width, y + height), 1)
    second = first + torch.stack((dx, dy, dx, dy), 1)
    pairs = torch.stack((first, second), 1).float()

    return pairs
