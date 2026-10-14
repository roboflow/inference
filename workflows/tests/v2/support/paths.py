"""Stable roots for tests that read files outside the test tree.

This is the only place in the V2 test tree that counts parent directories.
"""

from pathlib import Path

WORKFLOWS_ROOT = Path(__file__).resolve().parents[3]
CHECKOUT_ROOT = WORKFLOWS_ROOT.parent
