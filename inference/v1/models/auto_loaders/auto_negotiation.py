from typing import List

from inference.v1.runtime_introspection.core import x_ray_runtime_environment
from inference.v1.weights_providers.entities import ModelPackageMetadata


def negotiate_model_package(
    model_packages: List[ModelPackageMetadata],
) -> ModelPackageMetadata:
    runtime_xray = x_ray_runtime_environment()
