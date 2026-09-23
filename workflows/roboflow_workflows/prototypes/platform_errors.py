"""Exception classes Workflow blocks raise and catch around platform calls.

RELOCATED, not duplicated: `inference/core/exceptions.py` imports these names
back, so there is exactly ONE class object per name. That keeps `__name__`
(serialized as `error_type` / `inner_error_type`), the constructors, the
subclass hierarchy and every `isinstance` check byte-identical, which a
parallel hierarchy could not.

The names stay Roboflow-branded on purpose: the decontamination goal is import
independence, not vocabulary independence, and renaming them would change the
error payloads clients match on.

Keep this module dependency-free apart from `typing` - `inference.core.exceptions`
imports it, and anything else would create a cycle.
"""

from typing import Optional


class RoboflowAPIRequestError(Exception):
    pass


class RoboflowAPIUnsuccessfulRequestError(RoboflowAPIRequestError):
    pass


class RoboflowAPIForbiddenError(RoboflowAPIUnsuccessfulRequestError):
    pass


class FeatureDeprecatedError(Exception):
    """Raised when a removed/deprecated feature is invoked.

    Maps to HTTP 410 Gone via the standard error handler; surfaces as
    error_type="FeatureDeprecatedError" in InferencePipeline StatusUpdate
    payloads and as the inner_error of
    ClientCausedStepExecutionError(status_code=410) when raised from a
    workflow block.
    """

    def __init__(
        self,
        feature: str,
        *,
        removal_release: Optional[str] = None,
        replacement: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        self.feature = feature
        self.removal_release = removal_release
        self.replacement = replacement
        self.reason = reason
        public = f"Feature '{feature}' has been removed from inference."
        if reason:
            public += f" Reason: {reason}."
        if removal_release:
            public += f" Removed in {removal_release}."
        public += (
            " No drop-in replacement is provided; contact Roboflow if you "
            "require this capability."
        )
        if replacement:
            public += f" Closest replacement: {replacement}."
        self._public_message = public
        super().__init__(public)

    def get_structured_public_error_details(self) -> dict:
        return {
            "feature": self.feature,
            "removal_release": self.removal_release,
            "replacement": self.replacement,
            "reason": self.reason,
        }
