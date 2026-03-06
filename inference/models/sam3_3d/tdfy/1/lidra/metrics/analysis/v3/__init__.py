"""Metrics analysis v3 - cleaner architecture with base Processor pattern."""

from .processor.multi_trial import MultiTrialConfig, MultiTrialProcessor

__all__ = ["MultiTrialConfig", "MultiTrialProcessor"]