import inspect

import pytest

REQUIRED = object()

# Exact normalized signatures: (name, parameter kind, default VALUE or REQUIRED).
# Fill the table from the REAL methods when writing gateway.py, then freeze.
EXPECTED_GATEWAY_SIGNATURES = {
    "start": [],
    "shutdown": [],
    "ensure_loaded": [
        ("model_id", "POSITIONAL_OR_KEYWORD", REQUIRED),
        ("instance", "POSITIONAL_OR_KEYWORD", ""),
        ("api_key", "POSITIONAL_OR_KEYWORD", ""),
        ("device", "POSITIONAL_OR_KEYWORD", ""),
    ],
    "load": [
        ("model_id", "POSITIONAL_OR_KEYWORD", REQUIRED),
        ("api_key", "POSITIONAL_OR_KEYWORD", ""),
        ("timeout_s", "POSITIONAL_OR_KEYWORD", None),
        ("pinned", "POSITIONAL_OR_KEYWORD", True),
    ],
    "unload": [("model_id", "POSITIONAL_OR_KEYWORD", REQUIRED)],
    "infer": [
        ("model_id", "KEYWORD_ONLY", REQUIRED),
        ("image", "KEYWORD_ONLY", None),
        ("action", "KEYWORD_ONLY", None),
        ("instance", "KEYWORD_ONLY", ""),
        ("params", "KEYWORD_ONLY", None),
        ("request", "KEYWORD_ONLY", None),
    ],
    "stats": [],
    "interface": [("model_id", "POSITIONAL_OR_KEYWORD", REQUIRED)],
}


def _normalized(fn):
    return [
        (
            p.name,
            p.kind.name,
            REQUIRED if p.default is inspect.Parameter.empty else p.default,
        )
        for p in inspect.signature(fn).parameters.values()
        if p.name != "self"
    ]


def test_direct_gateway_satisfies_contract():
    from inference_server.gateway import ModelManagerGateway

    for name, expected in EXPECTED_GATEWAY_SIGNATURES.items():
        fn = getattr(ModelManagerGateway, name)
        assert inspect.iscoroutinefunction(fn), name
        assert _normalized(fn) == expected, name


class _Manager:
    def __init__(self):
        self.loaded = set()
        self.executor = None

    def __contains__(self, key):
        return key in self.loaded

    def load(self, key, api_key, **kwargs):
        self.loaded.add(key)


@pytest.mark.asyncio
async def test_ensure_loaded_reports_model_ready_for_fresh_and_present_models():
    from inference_server.gateway import ModelManagerGateway

    gateway = ModelManagerGateway(_Manager())

    fresh = await gateway.ensure_loaded("m")
    again = await gateway.ensure_loaded("m")

    assert fresh == ("model_ready",)
    assert again == ("model_ready",)


class _FailingManager(_Manager):
    def load(self, key, api_key, **kwargs):
        raise RuntimeError("weights download failed")


@pytest.mark.asyncio
async def test_failed_load_reports_error_and_code_before_the_optional_description():
    import json

    from inference_server.gateway import ModelManagerGateway

    gateway = ModelManagerGateway(_FailingManager())

    ensured = await gateway.ensure_loaded("m")
    loaded = await gateway.load("m")

    for result in (ensured, loaded):
        assert result[:2] == ("error", 5)
        assert set(result[2]) == {
            "error_type",
            "message",
            "help_url",
            "status_code",
            "restricted",
        }
        assert json.loads(json.dumps(result[2])) == result[2]


def test_last_load_failure_is_not_part_of_the_required_surface():
    assert "last_load_failure" not in EXPECTED_GATEWAY_SIGNATURES
