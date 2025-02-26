import os
import sys
import importlib
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory


@pytest.fixture(scope="session")
def builder_env_session():
    """
    A session-scoped fixture that:
      1. Creates a temporary directory.
      2. Sets MODEL_CACHE_DIR to that directory.
      3. Removes the 'inference.core.interfaces.http.builder.routes' module from sys.modules so we can re-import it fresh.
      4. Yields the temp directory path so tests can reference it if needed.
    """
    with TemporaryDirectory() as tmp_dir:
        os.environ["MODEL_CACHE_DIR"] = os.path.join(tmp_dir, "model_cache")

        module_name = "inference.core.interfaces.http.builder.routes"
        if module_name in sys.modules:
            del sys.modules[module_name]
        # Now any import of that module will happen fresh, picking up the new env

        yield tmp_dir  # yield so tests can read the path if desired

        # When the session ends, the temp dir is cleaned up automatically.
