"""Statistical functionals for metrics analysis."""

import numpy as np
from typing import Callable, Dict, List

# Type alias for clarity
StatisticalFunctional = Callable[[np.ndarray], float]


def mean(x: np.ndarray) -> float:
    """Arithmetic mean."""
    return float(np.mean(x))


def std(x: np.ndarray) -> float:
    """Standard deviation with Bessel's correction."""
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def percentile(p: float) -> StatisticalFunctional:
    """Create a percentile functional.

    Args:
        p: Percentile value (0-100)

    Returns:
        A functional that computes the p-th percentile
    """

    def _percentile(x: np.ndarray) -> float:
        return float(np.percentile(x, p))

    _percentile.__name__ = f"p{int(p)}"
    return _percentile


# Registry of available functionals
FUNCTIONALS: Dict[str, StatisticalFunctional] = {
    "mean": mean,
    "std": std,
    "min": lambda x: float(np.min(x)),
    "max": lambda x: float(np.max(x)),
    "p0": percentile(0),  # same as min
    "p5": percentile(5),
    "p25": percentile(25),
    "p50": percentile(50),  # median
    "p75": percentile(75),
    "p95": percentile(95),
    "p100": percentile(100),  # same as max
}


def get_functionals(names: List[str]) -> Dict[str, StatisticalFunctional]:
    """Get functionals by name.

    Args:
        names: List of functional names to retrieve

    Returns:
        Dictionary mapping names to functionals (only valid names included)
    """
    return {name: FUNCTIONALS[name] for name in names if name in FUNCTIONALS}
