from typing import Dict, Any


class PerSample:
    """
    Base class for per-sample metrics.
    """

    @staticmethod
    def evaluate(prediction: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
