import inspect

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
    ],
    "unload": [("model_id", "POSITIONAL_OR_KEYWORD", REQUIRED)],
    "infer": [
        ("model_id", "KEYWORD_ONLY", REQUIRED),
        ("image", "KEYWORD_ONLY", None),
        ("task", "KEYWORD_ONLY", None),
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
