"""Warning categories emitted by the camera, stream and stream-manager runtime.

`inference.core.warnings` imports these names back, so warning filters set on
the historical names keep matching. Keep this module free of imports.
"""


class InferenceExperimentalFeatureWarning(Warning):
    pass
