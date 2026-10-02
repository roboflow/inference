from typing import List, Optional, Type

import torch
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.roboflow.embedding.v1 import IMAGE_EMBEDDINGS
from roboflow_workflows.core_steps.models.roboflow.embedding.v1 import (
    BlockManifest as ListBlockManifest,
)
from roboflow_workflows.core_steps.models.roboflow.embedding.v1 import (
    EmbeddingModelBlockV1 as ListEmbeddingModelBlockV1,
)
from roboflow_workflows.core_steps.models.roboflow.embedding.v1 import (
    workflow_embedding_info,
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
        """Generate native local embeddings or rebuild remote JSON vectors.

        Args:
            data: Images or crops to embed in input order.
            model_id: Classification model version or pretrained alias.
            output_type: Feature vector or pre-activation logits.

        Returns:
            One tensor embedding and compact compatibility metadata per image.

        Raises:
            ValueError: If the model returns an incorrect embedding batch size.
        """
        if not data:
            return []

        if self._step_execution_mode is StepExecutionMode.LOCAL:
            if all(image.is_tensor_materialised() for image in data):
                images = [image.tensor_image for image in data]
                input_color_format = "rgb"
            else:
                images = [image.numpy_image for image in data]
                input_color_format = "bgr"

            self._model_manager.add_model(
                model_id=model_id,
                api_key=self._api_key,
                required_capabilities=[IMAGE_EMBEDDINGS],
                output_type=output_type,
            )
            response = self._model_manager.run_tensor_image_embeddings(
                model_id=model_id,
                images=images,
                input_color_format=input_color_format,
                api_key=self._api_key,
                output_type=output_type,
            )
            embeddings = response["embeddings"]
            if len(embeddings) != len(data):
                raise ValueError(
                    "Embedding response count does not match the image batch."
                )

            embeddings = embeddings.to(
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE, dtype=torch.float32
            )
            info = workflow_embedding_info(response["embedding_info"])
            results = [
                {"embedding": embedding, "embedding_info": info}
                for embedding in embeddings
            ]

            return results

        results = super().run(data=data, model_id=model_id, output_type=output_type)
        for result in results:
            result["embedding"] = torch.tensor(
                result["embedding"],
                dtype=torch.float32,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            )
        return results
