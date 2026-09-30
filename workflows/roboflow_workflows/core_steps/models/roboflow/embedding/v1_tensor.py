from typing import List, Type

import torch
from roboflow_workflows.core_steps.models.roboflow.embedding.v1 import (
    BlockManifest as ListBlockManifest,
)
from roboflow_workflows.core_steps.models.roboflow.embedding.v1 import (
    EmbeddingModelBlockV1 as ListEmbeddingModelBlockV1,
)
from roboflow_workflows.environment import WORKFLOWS_IMAGE_TENSOR_DEVICE
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_EMBEDDING_KIND,
)
from roboflow_workflows.execution_engine.entities.types import DICTIONARY_KIND
from roboflow_workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class BlockManifest(ListBlockManifest):
    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="embedding", kind=[TENSOR_NATIVE_EMBEDDING_KIND]),
            OutputDefinition(name="embedding_info", kind=[DICTIONARY_KIND]),
        ]


class EmbeddingModelBlockV1(ListEmbeddingModelBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        data: Batch[WorkflowImageData],
        model_id: str,
        output_type: str = "feature_vector",
    ) -> BlockResult:
        results = super().run(data=data, model_id=model_id, output_type=output_type)
        for result in results:
            result["embedding"] = torch.tensor(
                result["embedding"],
                dtype=torch.float32,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            )
        return results
