"""Integration test for Cosmos AnomalyGen on real weights.

Loads from a local package directory (the layout produced by
`development/cosmos3/pull_anomalygen_weights.py`) pointed to by an env var, so
it can run before the package is published to the weights provider:

    COSMOS_ANOMALYGEN_PACKAGE_DIR=checkpoints/packages/cosmos-anomalygen \
    COSMOS_ANOMALYGEN_ANOMALY_TYPE=tube+hole \
    python -m pytest tests/integration_tests/models/test_cosmos_anomalygen_predictions.py -m slow

The anomaly type must be one the package was trained on (first line of the
package's class_names.txt when the env var is unset). Requires the NVIDIA
paidf-anomalygen stack on the python path (a container built on the
`paidf-anomalygen:ga` base image) and a >=16GB GPU.

Once the package is registered, a conftest fixture downloading the published
zip should replace the env-var indirection (matching the other model suites).
"""

import os

import numpy as np
import pytest
import torch

PACKAGE_DIR = os.environ.get("COSMOS_ANOMALYGEN_PACKAGE_DIR")
CUDA_AVAILABLE = torch.cuda.is_available()


def _first_trained_anomaly_type(package_dir: str) -> str:
    configured = os.environ.get("COSMOS_ANOMALYGEN_ANOMALY_TYPE")
    if configured:
        return configured
    with open(os.path.join(package_dir, "class_names.txt")) as fp:
        return next(line.strip() for line in fp if line.strip())


@pytest.mark.slow
@pytest.mark.skipif(
    not PACKAGE_DIR or not CUDA_AVAILABLE,
    reason="COSMOS_ANOMALYGEN_PACKAGE_DIR not set or CUDA unavailable",
)
def test_cosmos_anomalygen_inpaints_a_defect() -> None:
    from inference_models.models.cosmos3.cosmos_anomalygen import CosmosAnomalyGen

    model = CosmosAnomalyGen.from_pretrained(PACKAGE_DIR, device=torch.device("cuda"))
    image = np.full((512, 512, 3), 180, dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[200:312, 200:312] = 255

    generated = model.generate(
        image=image,
        mask=mask,
        anomaly_type=_first_trained_anomaly_type(PACKAGE_DIR),
        seed=0,
    )

    assert len(generated) == 1
    assert generated[0].shape == image.shape
    assert generated[0].dtype == np.uint8
