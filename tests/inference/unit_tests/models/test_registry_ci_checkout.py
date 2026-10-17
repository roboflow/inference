"""Keep the historical registry oracle available in the unit-test checkout."""

from pathlib import Path

import yaml


def test_unit_checkout_includes_registry_oracle_history():
    """Require history for the independent inventory and disabled-flag oracle."""
    root = Path(__file__).resolve().parents[4]
    workflow = yaml.safe_load(
        (root / ".github/workflows/unit_tests_inference_x86.yml").read_text()
    )
    checkout = next(
        step
        for step in workflow["jobs"]["build-dev-test"]["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert (
        checkout.get("with", {}).get("fetch-depth", 1) == 0
    ), "Registry oracle tests execute a historical commit absent from a shallow checkout"
