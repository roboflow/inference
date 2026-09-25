"""Evaluating a restriction's configuration condition against THIS process.

``RuntimeRestriction.applies_to_configuration`` is written for the TARGET
deployment. Only one caller ever evaluates it against the host answering the
call: ``WorkflowBlockManifest.get_actual_restrictions()`` with
``ignore_environment_restrictions=False``.

Three rules this module exists to enforce:

* **The parsed package configuration is the only source.** Values come from
  ``roboflow_workflows.environment``, which binds the installed
  ``WorkflowsConfiguration`` to the historic ``UPPER_CASE`` names. No
  ``os.getenv``, no truthiness of a raw string.
* **The import is lazy.** Importing ``roboflow_workflows.environment`` FREEZES
  the process configuration (it binds its constants once, at import). This
  module must therefore never import it at module scope, because it is reached
  from ``roboflow_workflows.prototypes.block``, which the host imports long
  before it installs its configuration.
* **Unknown keys stay unknown.** Only the keys in ``EVALUABLE_CONFIGURATION_KEYS``
  are evaluated. Anything else - a misspelling, a key a newer block introduced,
  a name that happens to exist in the environment module for another reason -
  yields ``UNKNOWN``, never an accidental match and never a silent drop.

Diagnostics built from this module name configuration KEYS only. The host's
values never leave it.
"""

from enum import Enum
from typing import Any, Dict, FrozenSet, List, Tuple

from roboflow_workflows.execution_engine.entities.workload import RuntimeRestriction


class ConfigurationMatch(Enum):
    """The verdict on one restriction's configuration condition.

    MATCHES: every predicate was evaluable and held - the restriction applies
    to this process as far as configuration goes.
    INACTIVE: at least one predicate was evaluable and did NOT hold - the
    restriction definitively does not apply here.
    UNKNOWN: no predicate ruled it out, but at least one could not be
    evaluated - the restriction is kept and the uncertainty is reported.
    """

    MATCHES = "matches"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"


# Every configuration key a built-in declaration conditions on today. Each entry
# is the name of a module constant of `roboflow_workflows.environment`; the
# accompanying unit test pins that every one of them resolves.
EVALUABLE_CONFIGURATION_KEYS: FrozenSet[str] = frozenset(
    {
        # engine
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
        "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE",
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE",
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES",
        # image representation
        "ENABLE_TENSOR_DATA_REPRESENTATION",
        # per-model hosted endpoint gates
        "CORE_MODEL_GAZE_ENABLED",
        "CORE_MODEL_PE_ENABLED",
        "CORE_MODEL_SAM2_ENABLED",
        "CORE_MODEL_SAM3_ENABLED",
        "COSMOS3_ENABLED",
        "DEPTH_ESTIMATION_ENABLED",
        "FLORENCE2_ENABLED",
        "GLM_OCR_ENABLED",
        "LMM_ENABLED",
        "MOONDREAM2_ENABLED",
        "QWEN_2_5_ENABLED",
        "QWEN_3_5_ENABLED",
        "QWEN_3_ENABLED",
        "SAM3_3D_OBJECTS_ENABLED",
        "SMOLVLM2_ENABLED",
    }
)

_MISSING = object()


def read_configuration_value(key: str) -> Any:
    """The installed configuration's value for ``key``, or ``_MISSING``.

    Lazy import on purpose - see the module docstring. Reading the attribute
    through the module object (not a `from ... import`) is what lets a test
    that reloads `roboflow_workflows.environment` be seen here.
    """
    if key not in EVALUABLE_CONFIGURATION_KEYS:
        return _MISSING
    from roboflow_workflows import environment

    return getattr(environment, key, _MISSING)


def values_equal(declared: Any, actual: Any) -> bool:
    """Value equality that does not let ``True`` match ``1``.

    A declaration pins a flag (``False``), a mode (``"local"``) or a number;
    Python's ``1 == True`` would turn a numeric setting into a boolean match.
    """
    if isinstance(declared, bool) != isinstance(actual, bool):
        return False
    return declared == actual


def evaluate_configuration_condition(
    restriction: RuntimeRestriction,
) -> Tuple[ConfigurationMatch, List[str]]:
    """Judge one restriction's configuration predicates against this process.

    Returns the verdict and the keys that could not be evaluated (empty unless
    the verdict is ``UNKNOWN``). Predicates are ANDed: one evaluable predicate
    that does not hold makes the restriction ``INACTIVE`` whatever the other
    predicates say, because the restriction as a whole cannot apply.

    The restriction itself is never modified and its conditions are never
    stripped.
    """
    configuration: Dict[str, Any] = restriction.applies_to_configuration or {}
    unknown_keys: List[str] = []
    for key, declared in configuration.items():
        actual = read_configuration_value(key=key)
        if actual is _MISSING:
            unknown_keys.append(key)
            continue
        if not values_equal(declared=declared, actual=actual):
            return ConfigurationMatch.INACTIVE, []
    if unknown_keys:
        return ConfigurationMatch.UNKNOWN, unknown_keys
    return ConfigurationMatch.MATCHES, []
