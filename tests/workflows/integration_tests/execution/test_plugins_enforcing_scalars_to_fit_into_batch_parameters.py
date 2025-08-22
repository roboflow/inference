from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import AssumptionError
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
    "inputs": [
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
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
            "additional": "$inputs.confidence"
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
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
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
            "additional": "$inputs.confidence"
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


WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCERS = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
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
            "type": "MultiSIMDImageConsumerDecreasingDim",
            "name": "image_consumer",
            "images_x": "$steps.identity_simd.x",
            "images_y": "$steps.image_producer_y.image",
            "additional": "$inputs.confidence"
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
def test_workflow_with_multiple_scalar_producers_feeding_simd_consumer_decreasing_dim(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCERS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={})

    # then
    assert result[0]["shapes"] == "[192, 168, 3][220, 230, 3]"


WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCER_AND_BATCH_INPUT = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
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
            "type": "MultiSIMDImageConsumerDecreasingDim",
            "name": "image_consumer",
            "images_x": "$steps.identity_simd.x",
            "images_y": "$inputs.image",
            "additional": "$inputs.confidence"
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
def test_workflow_with_scalar_producer_and_batch_input_feeding_simd_consumer_decreasing_dim(
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
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_PRODUCER_AND_BATCH_INPUT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": [image_1, image_2]})

    # then
    assert result == [
        {"shapes": "[192, 168, 3][200, 100, 3]\n[192, 168, 3][300, 100, 3]"}
    ]


WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_BATCH_INPUTS_AT_DIM_1 = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
    "steps": [
        {
            "type": "IdentitySIMD",
            "name": "identity_simd",
            "x": "$inputs.image_1",
        },
        {
            "type": "Identity",
            "name": "identity_non_simd",
            "x": "$inputs.image_2",
        },
        {
            "type": "MultiSIMDImageConsumerDecreasingDim",
            "name": "image_consumer",
            "images_x": "$steps.identity_simd.x",
            "images_y": "$steps.identity_non_simd.x",
            "additional": "$inputs.confidence"
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd_2",
            "x": "$steps.image_consumer.shapes",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "shapes",
            "selector": "$steps.identity_simd_2.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_batched_inputs_at_dim_1_fed_into_consumer_decreasing_the_dimensionality(
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
    image_3 = np.zeros((400, 100, 3), dtype=np.uint8)
    image_4 = np.zeros((500, 100, 3), dtype=np.uint8)
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_BATCH_INPUTS_AT_DIM_1,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={
        "image_1": [image_1, image_2],
        "image_2": [image_3, image_4],
    })

    # then
    assert result == [
        {"shapes": "[200, 100, 3][400, 100, 3]\n[300, 100, 3][500, 100, 3]"}
    ]


WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_BATCH_INPUTS_AT_DIM_1_BOOSTING_DIM_AT_THE_END = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
    "steps": [
        {
            "type": "IdentitySIMD",
            "name": "identity_simd",
            "x": "$inputs.image_1",
        },
        {
            "type": "Identity",
            "name": "identity_non_simd",
            "x": "$inputs.image_2",
        },
        {
            "type": "MultiSIMDImageConsumerDecreasingDim",
            "name": "image_consumer",
            "images_x": "$steps.identity_simd.x",
            "images_y": "$steps.identity_non_simd.x",
            "additional": "$inputs.confidence"
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd_2",
            "x": "$steps.image_consumer.shapes",
        },
        {
            "type": "BoostDimensionality",
            "name": "dimensionality_boost",
            "x": "$steps.identity_simd_2.x"
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "shapes",
            "selector": "$steps.dimensionality_boost.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_batched_inputs_at_dim_1_fed_into_consumer_decreasing_the_dimensionality_and_boosting_scalar_dim_at_the_end(
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

    # when
    with pytest.raises(AssumptionError):
        # TESTING CURRENT LIMITATION OF EE - WE CANNOT HAVE A BLOCK THAT YIELDS NEW 1ST LEVEL
        # OF DIMENSIONALITY (WHICH IS DICTATED BY INPUTS)!
        _ = ExecutionEngine.init(
            workflow_definition=WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_BATCH_INPUTS_AT_DIM_1_BOOSTING_DIM_AT_THE_END,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
        )


WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_BATCH_INPUTS_AT_DIM_2 = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
    "steps": [
        {
            "type": "DoubleBoostDimensionality",
            "name": "dimensionality_boost",
            "x": "$inputs.image_1",
            "y": "$inputs.image_2",
        },
        {
            "type": "MultiSIMDImageConsumerDecreasingDim",
            "name": "image_consumer",
            "images_x": "$steps.dimensionality_boost.x",
            "images_y": "$steps.dimensionality_boost.y",
            "additional": "$inputs.confidence"
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd_2",
            "x": "$steps.image_consumer.shapes",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "shapes",
            "selector": "$steps.identity_simd_2.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_batched_inputs_at_dim_2_fed_into_consumer_decreasing_the_dimensionality(
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
    image_3 = np.zeros((400, 100, 3), dtype=np.uint8)
    image_4 = np.zeros((500, 100, 3), dtype=np.uint8)
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_BATCH_INPUTS_AT_DIM_2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={
        "image_1": [image_1, image_2],
        "image_2": [image_3, image_4],
    })

    # then
    assert result == [
        {"shapes": "[200, 100, 3][400, 100, 3]\n[200, 100, 3][400, 100, 3]"},
        {"shapes": "[300, 100, 3][500, 100, 3]\n[300, 100, 3][500, 100, 3]"}
    ]


WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_INPUTS_BOOSTING_DIM_AT_THE_END = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
    ],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_y",
        },
        {
            "type": "MultiSIMDImageConsumerDecreasingDim",
            "name": "image_consumer",
            "images_x": "$steps.image_producer_x.image",
            "images_y": "$steps.image_producer_y.image",
            "additional": "$inputs.confidence"
        },
        {
            "type": "IdentitySIMD",
            "name": "identity_simd_2",
            "x": "$steps.image_consumer.shapes",
        },
        {
            "type": "BoostDimensionality",
            "name": "dimensionality_boost",
            "x": "$steps.identity_simd_2.x"
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "shapes",
            "selector": "$steps.dimensionality_boost.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_scalar_inputs_fed_into_consumer_decreasing_the_dimensionality_and_boosting_scalar_dim_at_the_end(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CONSUMER_DECREASING_OUTPUT_DIM_FED_BY_SCALAR_INPUTS_BOOSTING_DIM_AT_THE_END,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={})

    # then
    assert results == [
        {"shapes": "[192, 168, 3][192, 168, 3]"},
        {"shapes": "[192, 168, 3][192, 168, 3]"}
    ]


WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_LIST_OF_SIMD_IMAGES = {
    "version": "1.1",
    "inputs": [],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (100, 100, 3)
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_y",
            "shape": (200, 200, 3),
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_z",
            "shape": (300, 300, 3),
        },
        {
            "type": "NonSIMDConsumerAcceptingList",
            "name": "image_consumer",
            "x": ["$steps.image_producer_x.image", "$steps.image_producer_y.image"],
            "y": ["$steps.image_producer_z.image"],
        },
        {
            "type": "BoostDimensionality",
            "name": "dimensionality_boost",
            "x": "$steps.image_consumer.x"
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.dimensionality_boost.x",
        },
        {
            "type": "JsonField",
            "name": "y",
            "selector": "$steps.image_consumer.y",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumers_accepting_list_of_scalar_selector(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_LIST_OF_SIMD_IMAGES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={})

    # then
    assert len(results) == 2, "Expected dim increase to happen"
    assert [i.numpy_image.shape for i in results[0]["x"]] == [(100, 100, 3), (200, 200, 3)]
    assert [i.numpy_image.shape for i in results[0]["y"]] == [(300, 300, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"]] == [(100, 100, 3), (200, 200, 3)]
    assert [i.numpy_image.shape for i in results[1]["y"]] == [(300, 300, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_LIST_OF_BATCH_IMAGES = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_2"},
        {"type": "WorkflowImage", "name": "image_3"},
    ],
    "steps": [
        {
            "type": "NonSIMDConsumerAcceptingList",
            "name": "image_consumer",
            "x": ["$inputs.image_1", "$inputs.image_2"],
            "y": ["$inputs.image_3"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
        {
            "type": "JsonField",
            "name": "y",
            "selector": "$steps.image_consumer.y",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumers_accepting_list_of_batch_selector(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_LIST_OF_BATCH_IMAGES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={
        "image_1": [np.zeros((100, 100, 3)), np.zeros((120, 120, 3))],
        "image_2": [np.zeros((200, 200, 3)), np.zeros((220, 220, 3))],
        "image_3": [np.zeros((300, 300, 3)), np.zeros((320, 320, 3))],
    })

    # then
    assert len(results) == 2
    assert [i.numpy_image.shape for i in results[0]["x"]] == [(100, 100, 3), (200, 200, 3)]
    assert [i.numpy_image.shape for i in results[0]["y"]] == [(300, 300, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"]] == [(120, 120, 3), (220, 220, 3)]
    assert [i.numpy_image.shape for i in results[1]["y"]] == [(320, 320, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_LIST_OF_BATCH_AND_SCALAR_IMAGES = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_3"},
    ],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingList",
            "name": "image_consumer",
            "x": ["$inputs.image_1", "$steps.image_producer_x.image"],
            "y": ["$inputs.image_3"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
        {
            "type": "JsonField",
            "name": "y",
            "selector": "$steps.image_consumer.y",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumers_accepting_list_of_batch_and_scalar_selector(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_LIST_OF_BATCH_AND_SCALAR_IMAGES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={
        "image_1": [np.zeros((100, 100, 3)), np.zeros((120, 120, 3))],
        "image_3": [np.zeros((300, 300, 3)), np.zeros((320, 320, 3))],
    })

    # then
    assert len(results) == 2
    assert [i.numpy_image.shape for i in results[0]["x"]] == [(100, 100, 3), (50, 50, 3)]
    assert [i.numpy_image.shape for i in results[0]["y"]] == [(300, 300, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"]] == [(120, 120, 3), (50, 50, 3)]
    assert [i.numpy_image.shape for i in results[1]["y"]] == [(320, 320, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_3"},
    ],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingDict",
            "name": "image_consumer",
            "x": {
                "a": "$inputs.image_1",
                "b": "$steps.image_producer_x.image",
                "c": "$inputs.image_3"
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumers_accepting_dict_of_batch_and_scalar_selector(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={
        "image_1": [np.zeros((100, 100, 3)), np.zeros((120, 120, 3))],
        "image_3": [np.zeros((300, 300, 3)), np.zeros((320, 320, 3))],
    })

    # then
    assert len(results) == 2
    assert [i.numpy_image.shape for i in results[0]["x"]] == [(100, 100, 3), (50, 50, 3), (300, 300, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"]] == [(120, 120, 3), (50, 50, 3), (320, 320, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_2 = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_3"},
    ],
    "steps": [
        {
            "type": "DoubleBoostDimensionality",
            "name": "dimensionality_boost",
            "x": "$inputs.image_1",
            "y": "$inputs.image_3",
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingDict",
            "name": "image_consumer",
            "x": {
                "a": "$steps.dimensionality_boost.x",
                "b": "$steps.image_producer_x.image",
                "c": "$steps.dimensionality_boost.y",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumers_accepting_dict_of_batch_and_scalar_selector_when_batch_at_dim_2(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={
        "image_1": [np.zeros((100, 100, 3)), np.zeros((120, 120, 3))],
        "image_3": [np.zeros((300, 300, 3)), np.zeros((320, 320, 3))],
    })

    # then
    assert len(results) == 2
    assert [i.numpy_image.shape for i in results[0]["x"][0]] == [(100, 100, 3), (50, 50, 3), (300, 300, 3)]
    assert [i.numpy_image.shape for i in results[0]["x"][1]] == [(100, 100, 3), (50, 50, 3), (300, 300, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"][0]] == [(120, 120, 3), (50, 50, 3), (320, 320, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"][1]] == [(120, 120, 3), (50, 50, 3), (320, 320, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_INCREASING_DIMENSIONALITY_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_2 = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_3"},
    ],
    "steps": [
        {
            "type": "DoubleBoostDimensionality",
            "name": "dimensionality_boost",
            "x": "$inputs.image_1",
            "y": "$inputs.image_3",
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingDictIncDim",
            "name": "image_consumer",
            "x": {
                "a": "$steps.dimensionality_boost.x",
                "b": "$steps.image_producer_x.image",
                "c": "$steps.dimensionality_boost.y",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumer_inc_dim_accepting_dict_of_batch_and_scalar_selector_when_batch_at_dim_2(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_INCREASING_DIMENSIONALITY_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={
        "image_1": [np.zeros((100, 100, 3)), np.zeros((120, 120, 3))],
        "image_3": [np.zeros((300, 300, 3)), np.zeros((320, 320, 3))],
    })

    # then
    assert len(results) == 2
    assert [i.numpy_image.shape for i in results[0]["x"][0][0]] == [(100, 100, 3), (50, 50, 3), (300, 300, 3)]
    assert [i.numpy_image.shape for i in results[0]["x"][0][1]] == [(100, 100, 3), (50, 50, 3), (300, 300, 3)]
    assert [i.numpy_image.shape for i in results[0]["x"][1][0]] == [(100, 100, 3), (50, 50, 3), (300, 300, 3)]
    assert [i.numpy_image.shape for i in results[0]["x"][1][1]] == [(100, 100, 3), (50, 50, 3), (300, 300, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"][0][0]] == [(120, 120, 3), (50, 50, 3), (320, 320, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"][0][1]] == [(120, 120, 3), (50, 50, 3), (320, 320, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"][1][0]] == [(120, 120, 3), (50, 50, 3), (320, 320, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"][1][1]] == [(120, 120, 3), (50, 50, 3), (320, 320, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_2 = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_3"},
    ],
    "steps": [
        {
            "type": "DoubleBoostDimensionality",
            "name": "dimensionality_boost",
            "x": "$inputs.image_1",
            "y": "$inputs.image_3",
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingDictDecDim",
            "name": "image_consumer",
            "x": {
                "a": "$steps.dimensionality_boost.x",
                "b": "$steps.image_producer_x.image",
                "c": "$steps.dimensionality_boost.y",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumer_dec_dim_accepting_dict_of_batch_and_scalar_selector_when_batch_at_dim_2(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={
        "image_1": [np.zeros((100, 100, 3)), np.zeros((120, 120, 3))],
        "image_3": [np.zeros((300, 300, 3)), np.zeros((320, 320, 3))],
    })

    # then
    assert len(results) == 2
    assert [i.numpy_image.shape for i in results[0]["x"][0]] == [(100, 100, 3), (100, 100, 3)]
    assert [i.numpy_image.shape for i in results[0]["x"][1]] == [(50, 50, 3), (50, 50, 3)]
    assert [i.numpy_image.shape for i in results[0]["x"][2]] == [(300, 300, 3), (300, 300, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"][0]] == [(120, 120, 3), (120, 120, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"][1]] == [(50, 50, 3), (50, 50, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"][2]] == [(320, 320, 3), (320, 320, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_1 = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_3"},
    ],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingDictDecDim",
            "name": "image_consumer",
            "x": {
                "a": "$inputs.image_1",
                "b": "$steps.image_producer_x.image",
                "c": "$inputs.image_3",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumer_dec_dim_accepting_dict_of_batch_and_scalar_selector_when_batch_at_dim_1(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_DICT_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_1,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={
        "image_1": [np.zeros((100, 100, 3)), np.zeros((120, 120, 3))],
        "image_3": [np.zeros((300, 300, 3)), np.zeros((320, 320, 3))],
    })

    # then
    assert len(results) == 1
    assert [i.numpy_image.shape for i in results[0]["x"][0]] == [(100, 100, 3), (120, 120, 3)]
    assert [i.numpy_image.shape for i in results[0]["x"][1]] == [(50, 50, 3), (50, 50, 3)]
    assert [i.numpy_image.shape for i in results[0]["x"][2]] == [(300, 300, 3), (320, 320, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_DICT_OF_SCALARS = {
    "version": "1.1",
    "inputs": [],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_y",
            "shape": (60, 60, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingDictDecDim",
            "name": "image_consumer",
            "x": {
                "a": "$steps.image_producer_x.image",
                "b": "$steps.image_producer_y.image",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumer_dec_dim_accepting_dict_of_batch_and_scalar_selectors(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_DICT_OF_SCALARS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={})

    # then
    assert len(results) == 1
    assert [i.numpy_image.shape for i in results[0]["x"][0]] == [(50, 50, 3)]
    assert [i.numpy_image.shape for i in results[0]["x"][1]] == [(60, 60, 3)]


#####################################

WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_LIST_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_2 = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_3"},
    ],
    "steps": [
        {
            "type": "DoubleBoostDimensionality",
            "name": "dimensionality_boost",
            "x": "$inputs.image_1",
            "y": "$inputs.image_3",
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingListDecDim",
            "name": "image_consumer",
            "x": [
                "$steps.dimensionality_boost.x",
                "$steps.image_producer_x.image",
                "$steps.dimensionality_boost.y",
            ],
            "y": "some-value"
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumer_dec_dim_accepting_list_of_batch_and_scalar_selector_when_batch_at_dim_2(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_LIST_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={
        "image_1": [np.zeros((100, 100, 3)), np.zeros((120, 120, 3))],
        "image_3": [np.zeros((300, 300, 3)), np.zeros((320, 320, 3))],
    })

    # then
    assert len(results) == 2
    assert [i.numpy_image.shape for i in results[0]["x"]] == [(100, 100, 3), (100, 100, 3), (50, 50, 3), (50, 50, 3), (300, 300, 3), (300, 300, 3)]
    assert [i.numpy_image.shape for i in results[1]["x"]] == [(120, 120, 3), (120, 120, 3), (50, 50, 3), (50, 50, 3), (320, 320, 3), (320, 320, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_LIST_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_1 = {
    "version": "1.1",
    "inputs": [
        {"type": "WorkflowImage", "name": "image_1"},
        {"type": "WorkflowImage", "name": "image_3"},
    ],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingListDecDim",
            "name": "image_consumer",
            "x": ["$inputs.image_1", "$steps.image_producer_x.image", "$inputs.image_3"],
            "y": "some-value"
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumer_dec_dim_accepting_list_of_batch_and_scalar_selector_when_batch_at_dim_1(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_LIST_OF_BATCH_AND_SCALAR_IMAGES_AT_DIM_1,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={
        "image_1": [np.zeros((100, 100, 3)), np.zeros((120, 120, 3))],
        "image_3": [np.zeros((300, 300, 3)), np.zeros((320, 320, 3))],
    })

    # then
    assert len(results) == 1
    assert [i.numpy_image.shape for i in results[0]["x"]] == [(100, 100, 3), (120, 120, 3), (50, 50, 3), (50, 50, 3), (300, 300, 3), (320, 320, 3)]


WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_LIST_OF_SCALARS = {
    "version": "1.1",
    "inputs": [],
    "steps": [
        {
            "type": "ImageProducer",
            "name": "image_producer_x",
            "shape": (50, 50, 3)
        },
        {
            "type": "ImageProducer",
            "name": "image_producer_y",
            "shape": (60, 60, 3)
        },
        {
            "type": "NonSIMDConsumerAcceptingListDecDim",
            "name": "image_consumer",
            "x": ["$steps.image_producer_x.image", "$steps.image_producer_y.image"],
            "y": "some-value"
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "x",
            "selector": "$steps.image_consumer.x",
        },
    ],
}


@mock.patch.object(blocks_loader, "get_plugin_modules")
def test_workflow_with_simd_consumer_dec_dim_accepting_list_of_batch_and_scalar_selectors(
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
        workflow_definition=WORKFLOW_WITH_SIMD_CUSTOMER_DECREASING_DIMENSIONALITY_ACCEPTING_LIST_OF_SCALARS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    results = execution_engine.run(runtime_parameters={})

    # then
    assert len(results) == 1
    assert [i.numpy_image.shape for i in results[0]["x"]] == [(50, 50, 3), (60, 60, 3)]
