"""Unit tests for launcher.py — launch_inprocess."""

from __future__ import annotations

import os

from inference_model_manager.model_manager import ModelManager
from inference_server.launcher import launch_inprocess

# ---------------------------------------------------------------------------
# launch_inprocess
# ---------------------------------------------------------------------------


class TestLaunchInprocess:
    def test_returns_model_manager(self) -> None:
        mm = launch_inprocess()
        assert isinstance(mm, ModelManager)
