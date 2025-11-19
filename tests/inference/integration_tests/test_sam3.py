import os

import pytest

from tests.inference.integration_tests.regression_test import bool_env

API_KEY = os.environ.get("API_KEY")
PORT = os.environ.get("PORT", 9001)
BASE_URL = os.environ.get("BASE_URL", "http://localhost")


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_SAM3_TESTS", True)),
    reason="Skipping SAM test",
)
def test():
    pass
