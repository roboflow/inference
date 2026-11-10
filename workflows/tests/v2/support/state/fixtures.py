"""Managed-state fixtures shared by the memory and Redis state suites.

Registered by ``unit/engine/state/conftest.py`` and
``integration/external/redis/conftest.py``.
"""

import uuid

import pytest


@pytest.fixture
def namespace() -> str:
    unique_namespace = f"test-{uuid.uuid4().hex}"

    return unique_namespace
