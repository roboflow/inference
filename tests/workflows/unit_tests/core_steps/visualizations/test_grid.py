import numpy as np

from inference.core.workflows.core_steps.visualizations.grid.v1 import (
    GridVisualizationBlockV1,
    GridVisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_grid_manifest_accepts_inline_list_of_image_selectors() -> None:
    # given a comparison use case: input image next to a model visualization
    manifest = GridVisualizationManifest.model_validate(
        {
            "type": "roboflow_core/grid_visualization@v1",
            "name": "grid",
            "images": ["$inputs.image", "$steps.depth_estimation.image"],
        }
    )

    # then the inline list of image selectors is preserved
    assert manifest.images == ["$inputs.image", "$steps.depth_estimation.image"]


def test_grid_manifest_coerces_legacy_single_selector_to_list() -> None:
    # given the pre-existing usage: a single selector to a list of images
    manifest = GridVisualizationManifest.model_validate(
        {
            "type": "roboflow_core/grid_visualization@v1",
            "name": "grid",
            "images": "$steps.buffer.output",
        }
    )

    # then it is coerced to a one-element list so old workflows keep working
    assert manifest.images == ["$steps.buffer.output"]


def test_grid_visualization_block_single() -> None:
    # given
    block = GridVisualizationBlockV1()

    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )

    output = block.run(images=[image], width=1000, height=1000)

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check that the output is the same as the input
    assert np.array_equal(
        output.get("image").numpy_image, np.zeros((1000, 1000, 3), dtype=np.uint8)
    )


def test_grid_visualization_block_flattens_mixed_image_and_list_inputs() -> None:
    # given a mix of a single image (image selector) and a list of images
    # (a LIST_OF_VALUES selector, e.g. Buffer output)
    block = GridVisualizationBlockV1()

    single = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )
    list_of_two = [
        WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        ),
        WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
        ),
    ]

    # when the entries are flattened, three images are laid out (2x2 grid)
    output = block.run(images=[single, list_of_two], width=400, height=400)

    assert output is not None
    assert output.get("image").numpy_image.shape == (400, 400, 3)


def test_grid_visualization_block_2x2() -> None:
    # given
    block = GridVisualizationBlockV1()

    # 1000x1000 black
    image_1 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((1000, 1000, 3), dtype=np.uint8),
    )
    # 1000x1000 white
    image_2 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.array([[[255, 255, 255]] * 1000] * 1000, dtype=np.uint8),
    )
    # 1000x1000 red
    image_3 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.array([[[255, 0, 0]] * 1000] * 1000, dtype=np.uint8),
    )
    # 1000x1000 green
    image_4 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.array([[[0, 255, 0]] * 1000] * 1000, dtype=np.uint8),
    )

    output = block.run(
        images=[image_1, image_2, image_3, image_4], width=400, height=400
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match params
    assert output.get("image").numpy_image.shape == (400, 400, 3)

    # check that each quadrant is the right color
    # top left: black
    assert np.array_equal(
        output.get("image").numpy_image[:200, :200, :],
        np.zeros((200, 200, 3), dtype=np.uint8),
    )
    # top right: white
    assert np.array_equal(
        output.get("image").numpy_image[:200, 200:, :],
        np.array([[[255, 255, 255]] * 200] * 200, dtype=np.uint8),
    )
    # bottom left: red
    assert np.array_equal(
        output.get("image").numpy_image[200:, :200, :],
        np.array([[[255, 0, 0]] * 200] * 200, dtype=np.uint8),
    )
    # bottom right: green
    assert np.array_equal(
        output.get("image").numpy_image[200:, 200:, :],
        np.array([[[0, 255, 0]] * 200] * 200, dtype=np.uint8),
    )
