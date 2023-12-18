import os.path

import pytest


@pytest.fixture(scope="function")
def text_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), "assets", "file_with_lines.txt")
