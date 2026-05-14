from inference.core.exceptions import FeatureDeprecatedError


class InferenceModelsGazeAdapter:
    """Deprecated. Raises FeatureDeprecatedError on instantiation.

    The cross-tree adapter that bridged the legacy `inference/` Gaze API
    onto `inference_models.model_pipelines.face_and_gaze_detection` is no
    longer wired — the new tree's `FaceAndGazeDetectionMPAndL2CS` pipeline
    and the underlying MediaPipe dependency have been removed.

    The stub class is kept so the registry adapter dispatch at
    `inference/models/utils.py` still resolves `("gaze", "l2cs")` and
    surfaces a clear deprecation error instead of an ImportError.
    """

    def __init__(self, *args, **kwargs):
        raise FeatureDeprecatedError(
            feature="Gaze (L2CS-Net) model via inference_models adapter",
            reason="MediaPipe dependency removed from inference.",
        )
