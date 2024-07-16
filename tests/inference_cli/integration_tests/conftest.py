import os.path

import pytest


os.environ["TELEMETRY_OPT_OUT"] = "True"


@pytest.fixture(scope="function")
def example_env_file_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "assets", "example.env")
    )
