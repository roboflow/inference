from typing import Dict, List, Set, Union

from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_all_defined_kinds,
)
from inference.core.workflows.execution_engine.v1.introspection.kinds_schemas_register import (
    KIND_TO_SCHEMA_REGISTER,
)


def discover_kinds_typing_hints(kinds_names: Set[str]) -> Dict[str, str]:
    all_defined_kinds = load_all_defined_kinds()
    return {
        kind.name: kind.serialised_data_type
        for kind in all_defined_kinds
        if kind.serialised_data_type is not None and kind.name in kinds_names
    }


def discover_kinds_schemas(kinds_names: Set[str]) -> Dict[str, Union[dict, List[dict]]]:
    return {
        name: schema
        for name, schema in KIND_TO_SCHEMA_REGISTER.items()
        if name in kinds_names
    }
