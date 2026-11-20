import pytest
from streamvision.stream.pipeline import InferencePipeline


@pytest.mark.parametrize(
    "video_reference, video_identifiers, message",
    [
        ("a.mp4", ["line-1", "line-2"], "video_identifiers has 2 entries"),
        (["a.mp4", "b.mp4"], ["line-1"], "video_identifiers has 1 entries"),
        (["a.mp4", "b.mp4"], ["line-1", ""], "must be non-empty strings"),
        (["a.mp4", "b.mp4"], ["line-1", 2], "must be non-empty strings"),
    ],
)
def test_init_with_workflow_rejects_invalid_video_identifiers(
    video_reference, video_identifiers, message
):
    with pytest.raises(ValueError, match=message):
        InferencePipeline.init_with_workflow(
            video_reference,
            workflow_specification={"version": "1.0"},
            workflow_init_parameters={},
            step_error_handler=None,
            video_identifiers=video_identifiers,
        )
