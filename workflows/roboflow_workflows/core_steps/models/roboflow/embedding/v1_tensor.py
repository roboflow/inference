from typing import List, Optional, Type

import torch
from roboflow_workflows.core_steps.models.roboflow.embedding.v1 import IMAGE_EMBEDDINGS
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
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    WorkflowBlockManifest,
    roboflow_platform_model,
)


class BlockManifest(ListBlockManifest):
    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="embedding", kind=[TENSOR_NATIVE_EMBEDDING_KIND]),
            OutputDefinition(name="embedding_info", kind=[DICTIONARY_KIND]),
        ]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        return [
            roboflow_platform_model(
                model_id=self.model_id,
                model_registration_kwargs={
                    "required_capabilities": [IMAGE_EMBEDDINGS],
                    "output_type": self.output_type,
                },
            )
        ]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.MODEL_INFERENCE]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


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
