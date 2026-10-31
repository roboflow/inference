import tempfile
from typing import Generator

import pytest


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


# --- aiohttp 3.14 / aioresponses compatibility shim --------------------------
# aiohttp 3.14.0 made `stream_writer` a required keyword-only argument of
# ClientResponse.__init__. aioresponses (<=0.7.8) builds its mock response
# without it, raising at construction time:
#   TypeError: ClientResponse.__init__() missing 1 required keyword-only
#   argument: 'stream_writer'
# The upstream fix is unmerged (pnuckowski/aioresponses#288) and unreleased, so
# we replicate it locally: default aioresponses' response class to a subclass
# that injects Mock(output_size=0) (the only attribute aiohttp reads at init).
# Signature-guarded, so it is a no-op on aiohttp < 3.14 and can be deleted once
# a fixed aioresponses ships.
import inspect as _inspect
from unittest.mock import Mock as _Mock

import aioresponses.core as _aioresponses_core
from aiohttp.client_reqrep import ClientResponse as _ClientResponse

_AIOHTTP_NEEDS_STREAM_WRITER = (
    "stream_writer" in _inspect.signature(_ClientResponse).parameters
)


class _CompatClientResponse(_ClientResponse):
    def __init__(self, *args, **kwargs):
        if _AIOHTTP_NEEDS_STREAM_WRITER and "stream_writer" not in kwargs:
            kwargs["stream_writer"] = _Mock(output_size=0)
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True)
def _patch_aioresponses_stream_writer(monkeypatch):
    # aioresponses._build_response defaults response_class to the module-global
    # ClientResponse; swap it for the compat subclass for the duration of each test.
    monkeypatch.setattr(_aioresponses_core, "ClientResponse", _CompatClientResponse)
