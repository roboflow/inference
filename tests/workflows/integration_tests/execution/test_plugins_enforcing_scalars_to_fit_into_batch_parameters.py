from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import StepInputDimensionalityError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection import blocks_loader

WORKFLOW_IMAGE_PRODUCER_SINGLE_IMAGE_SIMD_CONSUMER = {
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
            "images": "$steps.image_producer.image",
        },
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
        workflow_definition=WORKFLOW_IMAGE_PRODUCER_SINGLE_IMAGE_SIMD_CONSUMER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert result == [{"shapes": "[192, 168, 3]"}]


WORKFLOW_IMAGE_PRODUCER_SINGLE_IMAGE_NON_SIMD_CONSUMER = {
    "version": "1.1",
    "inputs": [],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer",
        },
        {
            "type": "ImageConsumerNonSIMD",
            "name": "image_consumer",
            "images": "$steps.image_producer.image",
        },
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
def test_workflow_producing_image_and_consuming_it_in_block_accepting_single_non_simd_input(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
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
        workflow_definition=WORKFLOW_IMAGE_PRODUCER_SINGLE_IMAGE_NON_SIMD_CONSUMER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert result == [{"shapes": "[192, 168, 3]"}]


WORKFLOW_SINGLE_IMAGE_SIMD_CONSUMER_FROM_INPUT = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {"type": "ImageConsumer", "name": "image_consumer", "images": "$inputs.image"}
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
def test_workflow_consuming_input_image_in_block_accepting_single_non_simd_input(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
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
        workflow_definition=WORKFLOW_SINGLE_IMAGE_SIMD_CONSUMER_FROM_INPUT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    image = np.zeros((240, 230, 3), dtype=np.uint8)

    # when
    result = execution_engine.run(runtime_parameters={"image": image})

    # then
    assert result == [{"shapes": "[240, 230, 3]"}]


WORKFLOW_IMAGE_PRODUCER_MULTIPLE_IMAGES_SIMD_CONSUMER = {
    "version": "1.1",
    "inputs": [],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
        },
        {"type": "ImageProducer", "name": "image_producer_y", "shape": (240, 230, 3)},
        {
            "type": "MultiSIMDImageConsumer",
            "name": "image_consumer",
            "images_x": "$steps.image_producer_x.image",
            "images_y": "$steps.image_producer_y.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "metadata",
            "selector": "$steps.image_consumer.metadata",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_image_consumers_consuming_images_generated_by_image_producers_outputting_scalar_images(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
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
        workflow_definition=WORKFLOW_IMAGE_PRODUCER_MULTIPLE_IMAGES_SIMD_CONSUMER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert result == [{"metadata": "[192, 168, 3][240, 230, 3]"}]


WORKFLOW_IMAGE_PRODUCER_AND_INPUT_IMAGES_COMBINED_WITH_MULTIPLE_IMAGES_SIMD_CONSUMER = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
        },
        {
            "type": "MultiSIMDImageConsumer",
            "name": "image_consumer",
            "images_x": "$steps.image_producer_x.image",
            "images_y": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "metadata",
            "selector": "$steps.image_consumer.metadata",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_image_consumers_consuming_images_generated_by_image_producer_and_input_images_batch(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
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
        workflow_definition=WORKFLOW_IMAGE_PRODUCER_AND_INPUT_IMAGES_COMBINED_WITH_MULTIPLE_IMAGES_SIMD_CONSUMER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    input_images = [
        np.zeros((192, 192, 3), dtype=np.uint8),
        np.zeros((200, 192, 3), dtype=np.uint8),
        np.zeros((300, 192, 3), dtype=np.uint8),
    ]
    # when
    result = execution_engine.run(runtime_parameters={"image": input_images})

    # then
    assert result == [
        {"metadata": "[192, 168, 3][192, 192, 3]"},
        {"metadata": "[192, 168, 3][200, 192, 3]"},
        {"metadata": "[192, 168, 3][300, 192, 3]"},
    ]


WORKFLOW_IMAGE_PRODUCER_AND_STEP_OUTPUT_IMAGES_COMBINED_WITH_MULTIPLE_IMAGES_SIMD_CONSUMER = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd",
            "x": "$inputs.image",
        },
        {
            "type": "MultiSIMDImageConsumer",
            "name": "image_consumer",
            "images_x": "$steps.image_producer_x.image",
            "images_y": "$steps.identity_simd.x",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "metadata",
            "selector": "$steps.image_consumer.metadata",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_image_consumers_consuming_images_generated_by_image_producer_and_another_simd_block(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
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
        workflow_definition=WORKFLOW_IMAGE_PRODUCER_AND_STEP_OUTPUT_IMAGES_COMBINED_WITH_MULTIPLE_IMAGES_SIMD_CONSUMER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    input_images = [
        np.zeros((192, 192, 3), dtype=np.uint8),
        np.zeros((200, 192, 3), dtype=np.uint8),
        np.zeros((300, 192, 3), dtype=np.uint8),
    ]
    # when
    result = execution_engine.run(runtime_parameters={"image": input_images})

    # then
    assert result == [
        {"metadata": "[192, 168, 3][192, 192, 3]"},
        {"metadata": "[192, 168, 3][200, 192, 3]"},
        {"metadata": "[192, 168, 3][300, 192, 3]"},
    ]


WORKFLOW_WITH_SCALAR_MULTI_IMAGE_CONSUMER_FED_BY_SCALAR_PRODUCERS = {
    "version": "1.1",
    "inputs": [],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd",
            "x": "$steps.image_producer_x.image",
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_y",
            "shape": (220, 230, 3),
        },
        {
            "type": "MultiImageConsumer",
            "name": "image_consumer",
            "images_x": "$steps.identity_simd.x",
            "images_y": "$steps.image_producer_y.image",
        },
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
def test_workflow_with_multiple_scalar_producers_feeding_data_into_scalar_consumer(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
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
        workflow_definition=WORKFLOW_WITH_SCALAR_MULTI_IMAGE_CONSUMER_FED_BY_SCALAR_PRODUCERS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert result == [{"shapes": "[192, 168, 3][220, 230, 3]"}]


WORKFLOW_WITH_SCALAR_MULTI_IMAGE_CONSUMER_FED_BY_SCALAR_PRODUCER_AND_BATCH_INPUT = {
    "version": "1.1",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_y",
            "shape": (220, 230, 3),
        },
        {
            "type": "MultiImageConsumer",
            "name": "image_consumer",
            "images_x": "$inputs.image",
            "images_y": "$steps.image_producer_y.image",
        },
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
def test_workflow_with_scalar_producer_and_batch_input_feeding_data_into_scalar_consumer(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
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
        workflow_definition=WORKFLOW_WITH_SCALAR_MULTI_IMAGE_CONSUMER_FED_BY_SCALAR_PRODUCER_AND_BATCH_INPUT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    image = np.zeros((200, 400, 3), dtype=np.uint8)

    # when
    result = execution_engine.run(runtime_parameters={"image": image})

    # then
    assert result == [{"shapes": "[200, 400, 3][220, 230, 3]"}]


WORKFLOW_WITH_NON_SIMD_CONSUMER_RAISING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCERS = {
    "version": "1.1",
    "inputs": [],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd",
            "x": "$steps.image_producer_x.image",
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_y",
            "shape": (220, 230, 3),
        },
        {
            "type": "MultiImageConsumerRaisingDim",
            "name": "image_consumer",
            "images_x": "$steps.identity_simd.x",
            "images_y": "$steps.image_producer_y.image",
        },
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
def test_workflow_with_multiple_scalar_producers_feeding_non_simd_consumer_raising_dim(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_image_producer"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # then
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_SIMD_CONSUMER_RAISING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCERS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert result == [{"shapes": "[192, 168, 3][220, 230, 3]"}]


WORKFLOW_WITH_NON_SIMD_CONSUMER_RAISING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCER_AND_BATCH_INPUT = {
    "version": "1.1",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd",
            "x": "$steps.image_producer_x.image",
        },
        {
            "type": "MultiImageConsumerRaisingDim",
            "name": "image_consumer",
            "images_x": "$steps.identity_simd.x",
            "images_y": "$inputs.image",
        },
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
def test_workflow_with_scalar_producer_and_batch_input_feeding_non_simd_consumer_raising_dim(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_image_producer"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # then
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_SIMD_CONSUMER_RAISING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCER_AND_BATCH_INPUT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    image_1 = np.zeros((200, 100, 3), dtype=np.uint8)
    image_2 = np.zeros((300, 100, 3), dtype=np.uint8)

    # when
    result = execution_engine.run(runtime_parameters={"image": [image_1, image_2]})

    # then
    assert result == [
        {"shapes": ["[192, 168, 3][200, 100, 3]"]},
        {"shapes": ["[192, 168, 3][300, 100, 3]"]},
    ]


WORKFLOW_WITH_NON_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCERS = {
    "version": "1.1",
    "inputs": [],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd",
            "x": "$steps.image_producer_x.image",
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_y",
            "shape": (220, 230, 3),
        },
        {
            "type": "MultiNonSIMDImageConsumerDecreasingDim",
            "name": "image_consumer",
            "images_x": "$steps.identity_simd.x",
            "images_y": "$steps.image_producer_y.image",
        },
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
def test_workflow_with_multiple_scalar_producers_feeding_non_simd_consumer_decreasing_dim(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
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
        workflow_definition=WORKFLOW_WITH_NON_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCERS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert result[0]["shapes"] == "[192, 168, 3][220, 230, 3]"


WORKFLOW_WITH_NON_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCER_AND_BATCH_INPUT = {
    "version": "1.1",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd",
            "x": "$steps.image_producer_x.image",
        },
        {
            "type": "MultiNonSIMDImageConsumerDecreasingDim",
            "name": "image_consumer",
            "images_x": "$steps.identity_simd.x",
            "images_y": "$inputs.image",
        },
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
def test_workflow_with_scalar_producer_and_batch_input_feeding_non_simd_consumer_decreasing_dim(
    get_plugin_modules_mock: MagicMock,
    model_manager: ModelManager,
) -> None:
    # given
    get_plugin_modules_mock.return_value = [
        "tests.workflows.integration_tests.execution.stub_plugins.plugin_image_producer"
    ]
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    image_1 = np.zeros((200, 100, 3), dtype=np.uint8)
    image_2 = np.zeros((300, 100, 3), dtype=np.uint8)

    # then
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_NON_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCER_AND_BATCH_INPUT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # then
    result = execution_engine.run(runtime_parameters={"image": [image_1, image_2]})

    # then
    assert result == [
        {"shapes": "[192, 168, 3][200, 100, 3]\n[192, 168, 3][300, 100, 3]"}
    ]
