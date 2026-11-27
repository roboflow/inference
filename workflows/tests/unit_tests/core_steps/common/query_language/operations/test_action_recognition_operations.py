from roboflow_workflows.execution_engine.core import ExecutionEngine


def test_property_definition_extracts_action_recognition_result_fields() -> None:
    fields = ["timeline", "analyzed_from_frame", "analyzed_until_frame", "error_status"]
    workflow = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "results",
                "kind": ["action_recognition_result"],
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/property_definition@v1",
                "name": f"extract_{field}",
                "data": "$inputs.results",
                "operations": [
                    {"type": "ActionRecognitionPropertyExtract", "property_name": field}
                ],
            }
            for field in fields
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": field,
                "selector": f"$steps.extract_{field}.output",
            }
            for field in fields
        ],
    }
    walk = {"class": "walk", "class_id": 0, "start_frame_idx": 14, "end_frame_idx": 18}
    completed = {
        "timeline": [walk],
        "analyzed_from_frame": 12,
        "analyzed_until_frame": 21,
        "error_status": "A later window was skipped.",
    }
    warmup = {
        "timeline": [],
        "analyzed_from_frame": None,
        "analyzed_until_frame": None,
        "error_status": "",
    }

    engine = ExecutionEngine.init(workflow_definition=workflow)
    try:
        outputs = engine.run(
            runtime_parameters={"results": [warmup, completed]}, serialize_results=True
        )
    finally:
        engine.close()

    assert outputs == [warmup, completed]
