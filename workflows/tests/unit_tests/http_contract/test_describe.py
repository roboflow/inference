def test_describe_workflow_interface_reports_kinds():
    from roboflow_workflows.http_contract.describe import describe_workflow_interface

    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [],
        "outputs": [{"type": "JsonField", "name": "out", "selector": "$inputs.image"}],
    }
    described = describe_workflow_interface(definition)
    assert described.inputs["image"] == ["*"] and "image" in described.kinds_schemas


def test_describe_workflows_blocks_lists_core_blocks():
    from roboflow_workflows.http_contract.describe import describe_workflows_blocks

    description = describe_workflows_blocks()
    assert any(
        b.manifest_type_identifier == "roboflow_core/roboflow_object_detection_model@v2"
        for b in description.blocks
    )
    assert (
        description.dynamic_block_definition_schema["title"] == "DynamicBlockDefinition"
    )
