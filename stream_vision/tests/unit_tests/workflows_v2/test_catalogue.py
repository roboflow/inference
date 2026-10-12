"""The optional video module participates in ordinary V2 plugin discovery."""

from roboflow_workflows.execution_engine.v2 import Catalogue, compile_workflow
from streamvision.workflows_v2 import (
    WORKFLOWS_V2_SOURCES,
    VideoStream,
    VideoStreamSet,
    create_catalogue,
)


def test_video_catalogue_discovery_compiles_without_opening_sources():
    catalogue = Catalogue.from_modules(
        [
            "roboflow_workflows.execution_engine.v2.blocks",
            "streamvision.workflows_v2",
        ]
    )
    plan = compile_workflow(
        {
            "version": "2.0",
            "sources": [
                {
                    "type": "video/stream@v1",
                    "name": "camera",
                    "reference": "unopened-local-file.mp4",
                }
            ],
            "steps": [],
            "outputs": [
                {
                    "type": "OutputGroup",
                    "name": "frames",
                    "anchor": "$sources.camera.image",
                    "outputs": [
                        {"name": "image", "selector": "$sources.camera.image"}
                    ],
                }
            ],
        },
        catalogue=catalogue,
    )

    assert plan.is_active
    assert WORKFLOWS_V2_SOURCES == (VideoStream, VideoStreamSet)
    assert set(create_catalogue().source_types) == {
        "video/stream@v1",
        "video/stream_set@v1",
    }
