import pytest

from inference.core.workflows.execution_engine.core import ExecutionEngine


@pytest.mark.parametrize("binding", ["static", "runtime"])
@pytest.mark.parametrize(
    ("polygon", "reference_size", "expected"),
    [
        pytest.param(
            [[1.25, 2.25], [11.75, 2.25], [11.75, 7.75], [1.25, 7.75]],
            "21,11",
            [{"width": 20.0, "height": 10.0, "longer": 20.0, "shorter": 10.0}],
            id="fractional",
        ),
        pytest.param(
            [[1, 2], [11, 2], [11, 7], [1, 7]],
            "20,10",
            [{"width": 20.0, "height": 10.0, "longer": 20.0, "shorter": 10.0}],
            id="integer",
        ),
        pytest.param([[1, 1], [2, 2], [3, 3]], "20,10", None, id="collinear"),
        pytest.param(
            [
                [0.25, 0.5],
                [9.84807753012208e19, 1.7364817766693034e19],
                [8.111595753452777e19, 1.1584559306791382e20],
                [-1.7364817766693034e19, 9.84807753012208e19],
            ],
            "20,10",
            None,
            id="overflow",
        ),
    ],
)
def test_size_measurement_uses_reference_polygon(
    binding: str, polygon: list, reference_size: str, expected: list | None
) -> None:
    """Calibrate Size Measurement from a literal or runtime polygon.

    Args:
        binding (str): Where the workflow receives the polygon.
        polygon (list): Reference vertices in pixel coordinates.
        reference_size (str): The width and height of the reference in real units.
        expected (list | None): Expected dimensions of the detected object.
    """
    inputs = [
        {
            "type": "WorkflowBatchInput",
            "name": "objects",
            "kind": ["object_detection_prediction"],
            "dimensionality": 1,
        }
    ]
    runtime_parameters = {
        "objects": [
            {
                "image": {"width": 64, "height": 64},
                "predictions": [
                    {
                        "x": 15,
                        "y": 12.5,
                        "width": 10,
                        "height": 5,
                        "class": "package",
                        "class_id": 0,
                        "confidence": 0.9,
                    }
                ],
            }
        ]
    }
    reference = polygon
    if binding == "runtime":
        inputs.append({"type": "WorkflowParameter", "name": "polygon"})
        runtime_parameters["polygon"] = polygon
        reference = "$inputs.polygon"
    workflow = {
        "version": "1.0",
        "inputs": inputs,
        "steps": [
            {
                "type": "roboflow_core/size_measurement@v1",
                "name": "measure",
                "reference_predictions": reference,
                "reference_dimensions": reference_size,
                "object_predictions": "$inputs.objects",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "dimensions",
                "selector": "$steps.measure.dimensions",
            }
        ],
    }
    engine = ExecutionEngine.init(workflow_definition=workflow, init_parameters={})

    try:
        result = engine.run(runtime_parameters=runtime_parameters)
    finally:
        engine.close()

    assert result == [{"dimensions": expected}]
