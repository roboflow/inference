"""Pytest configuration for introspection tests."""
import pytest

from inference.core.workflows.execution_engine.introspection import blocks_loader
from inference.core.workflows.execution_engine.introspection import schema_parser
from inference.core.workflows.execution_engine.v1.compiler import syntactic_parser


@pytest.fixture(autouse=True)
def clear_all_caches():
    """Clear all LRU caches before each test to ensure test isolation."""
    blocks_loader.clear_caches()
    schema_parser.clear_cache()
    syntactic_parser.clear_cache()
    yield
    # Also clear after test in case test modified state
    blocks_loader.clear_caches()
    schema_parser.clear_cache()
    syntactic_parser.clear_cache()
