"""Compatibility package for the Workflows plugin of inference 1.6.1-1.7.2.

Its submodules resolve to the Roboflow-platform blocks in
``roboflow_workflows.core_steps.{sinks,integrations}.roboflow``.
"""

# Server configuration must be installed before an aliased block module loads.
import inference.core  # noqa: F401  isort: skip
from inference._workflows_compat import install as _install_workflows_compat

_install_workflows_compat()
