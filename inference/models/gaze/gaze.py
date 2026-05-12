from inference.core.exceptions import FeatureDeprecatedError


class Gaze:
    """Deprecated. Raises FeatureDeprecatedError on instantiation.

    The legacy registry entry under ("gaze", "l2cs") still resolves to
    this class so model-id lookups return a clear deprecation error rather
    than KeyError. The underlying L2CS-Net + MediaPipe implementation has
    been removed from inference.
    """

    def __init__(self, *args, **kwargs):
        raise FeatureDeprecatedError(
            feature="Gaze (L2CS-Net) model",
            reason="MediaPipe dependency removed from inference.",
        )
