"""Header building and secure-gateway URL wrapping are injected, not local.

`wrap_url` is a security control: it routes outbound calls through the
configured SECURE_GATEWAY proxy. Reimplementing it inside a block would
silently bypass the gateway.
"""

import ast
import importlib
import pathlib
from unittest import mock

import numpy as np
import pytest

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from tests.workflows.unit_tests.prototypes.platform_client_double import (
    RecordingPlatformClient,
)

PREFIX = "inference.core.workflows.core_steps.models.foundation."
BLOCKS = [
    ("seg_preview.v1", "SegPreviewBlockV1"),
    ("seg_preview.v1_tensor", "SegPreviewBlockV1"),
    ("segment_anything3.v1", "SegmentAnything3BlockV1"),
    ("segment_anything3.v1_tensor", "SegmentAnything3BlockV1"),
    ("segment_anything3.v2", "SegmentAnything3BlockV2"),
    ("segment_anything3.v2_tensor", "SegmentAnything3BlockV2"),
    ("segment_anything3.v3", "SegmentAnything3BlockV3"),
    ("segment_anything3.v3_tensor", "SegmentAnything3BlockV3"),
    ("segment_anything3_interactive.v1", "SegmentAnything3InteractiveBlockV1"),
    ("segment_anything3_interactive.v1_tensor", "SegmentAnything3InteractiveBlockV1"),
    ("segment_anything2_video.v1", "SegmentAnything2VideoBlockV1"),
    ("segment_anything2_video.v1_tensor", "SegmentAnything2VideoBlockV1"),
    ("segment_anything3_video.v1", "SegmentAnything3VideoBlockV1"),
    ("segment_anything3_video.v1_tensor", "SegmentAnything3VideoBlockV1"),
]
WORKFLOWS_ROOT = (
    pathlib.Path(__file__).resolve().parents[6] / "inference" / "core" / "workflows"
)
BANNED_SYMBOLS = {
    "build_roboflow_api_headers",
    "get_extra_weights_provider_headers",
    "wrap_url",
}


@pytest.mark.parametrize("module_suffix,class_name", BLOCKS)
def test_block_declares_the_platform_client_init_parameter(module_suffix, class_name):
    module = importlib.import_module(PREFIX + module_suffix)
    assert "platform_client" in getattr(module, class_name).get_init_parameters()


@pytest.mark.parametrize("module_suffix,class_name", BLOCKS)
def test_block_stores_the_injected_client(module_suffix, class_name):
    sentinel = RecordingPlatformClient()
    block_class = getattr(importlib.import_module(PREFIX + module_suffix), class_name)
    kwargs = {name: None for name in block_class.get_init_parameters()}
    kwargs["platform_client"] = sentinel
    assert block_class(**kwargs)._platform_client is sentinel


def test_no_workflows_module_imports_the_header_or_url_helpers() -> None:
    """Symbol-specific: `inference.core.roboflow_api` still has legitimate
    importers until Task 9.6 (`block_scaffolding`, `reference_resolution`), so
    only these three names are banned here. Task 9.6 raises the ban to the
    whole module.
    """
    offenders = []
    for path in WORKFLOWS_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "inference.core.roboflow_api",
                "inference.core.utils.url_utils",
            }:
                for alias in node.names:
                    if alias.name in BANNED_SYMBOLS:
                        offenders.append(f"{path}:{node.lineno} {alias.name}")
    assert not offenders, offenders


def test_headers_and_url_go_through_the_injected_client(monkeypatch) -> None:
    """Behaviour, not declaration: `run_via_request` must call the port.

    `run_via_request(images, class_names, threshold)` (segment_anything3/v1.py:408)
    base64-encodes each image, builds `{"Content-Type": ...}` (+ the two
    internal-service headers when the env sets them - pinned empty here),
    passes them through `build_roboflow_api_headers`, POSTs to
    `wrap_url(f"{API_BASE_URL}/inferenceproxy/seg-preview?api_key={api_key}")`
    and converts `prompt_results`. An 8x8 black image and an empty result
    list are enough to drive the whole path. Executed GREEN against the
    codemod output while writing this plan (evidence E17).
    """
    from inference.core.workflows.core_steps.models.foundation.segment_anything3 import (
        v1,
    )

    monkeypatch.setattr(v1, "ROBOFLOW_INTERNAL_SERVICE_NAME", None)
    monkeypatch.setattr(v1, "ROBOFLOW_INTERNAL_SERVICE_SECRET", None)
    client = RecordingPlatformClient(wrap_prefix="https://gateway.local/proxy?url=")
    kwargs = {name: None for name in v1.SegmentAnything3BlockV1.get_init_parameters()}
    kwargs["api_key"] = "rf-key"
    kwargs["platform_client"] = client
    block = v1.SegmentAnything3BlockV1(**kwargs)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=np.zeros((8, 8, 3), dtype=np.uint8),
    )

    response = mock.MagicMock()
    response.json.return_value = {"prompt_results": []}
    response.raise_for_status.return_value = None
    with mock.patch.object(v1, "requests") as requests_mock:
        requests_mock.post.return_value = response
        result = block.run_via_request(
            images=Batch.init(content=[image], indices=[(0,)]),
            class_names=["cat"],
            threshold=0.5,
        )

    raw_url = f"{v1.API_BASE_URL}/inferenceproxy/seg-preview?api_key=rf-key"
    assert client.headers_calls == [{"Content-Type": "application/json"}]
    assert client.wrapped == [raw_url]
    assert requests_mock.post.call_args.args[0] == (
        "https://gateway.local/proxy?url=" + raw_url
    )
    assert requests_mock.post.call_args.kwargs["headers"] == {
        "X-Test": "1",
        "Content-Type": "application/json",
    }
    assert len(result) == 1 and len(result[0]["predictions"]) == 0
