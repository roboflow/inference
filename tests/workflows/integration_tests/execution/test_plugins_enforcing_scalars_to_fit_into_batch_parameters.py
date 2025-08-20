from unittest import mock
from unittest.mock import MagicMock

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader

WORKFLOW_PROCESSING_VIDEO_METADATA = {
    "version": "1.1",
    "inputs": [],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer",
        },
        {
            "type": "ImageConsumer",
            "name": "image_consumer",
            "images": "$steps.image_producer.image"
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "shapes",
            "selector": "$steps.image_consumer.shapes",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_producing_image_and_consuming_it_in_block_accepting_single_batch_input(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    """
    In this test scenario, we verify compatibility of new input type (WorkflowVideoMetadata)
    with Workflows compiler and execution engine.
    """
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_image_producer"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_PROCESSING_VIDEO_METADATA,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={}
    )

    # then
    print(result)
    raise Exception()
