import os
import sys
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture(scope="session")
def builder_env_session():
    """
    Creates a temporary directory and sets the MODEL_CACHE_DIR environment
    variable. Removes the routes module from sys.modules so it reloads with the new env.
    """
    with TemporaryDirectory() as tmp_dir:
        os.environ["MODEL_CACHE_DIR"] = os.path.join(tmp_dir, "model_cache")
        module_name = "inference.core.interfaces.http.builder.routes"
        if module_name in sys.modules:
            del sys.modules[module_name]
        yield tmp_dir
