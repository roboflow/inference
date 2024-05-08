import os
from datetime import datetime
from typing import Optional

import networkx as nx

from inference.core.utils.environment import str2bool
from inference.core.utils.file_system import ensure_parent_dir_exists

DUMP_EXECUTION_GRAPH_ENV = "DUMP_EXECUTION_GRAPH"
WORKFLOWS_DEBUG_DIR_ENV = "WORKFLOWS_DEBUG_DIR_ENV"
DEFAULT_WORKFLOWS_DEBUG_DIR = "/tmp/workflows"


def dump_execution_graph(
    execution_graph: nx.DiGraph,
    path: Optional[str] = None,
) -> Optional[str]:
    if path is None:
        if str2bool(os.getenv(DUMP_EXECUTION_GRAPH_ENV, "False")) is False:
            return None
        file_name_infix = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        path = os.path.join(
            os.getenv(WORKFLOWS_DEBUG_DIR_ENV, DEFAULT_WORKFLOWS_DEBUG_DIR),
            f"workflow_graph_{file_name_infix}.dot",
        )
    ensure_parent_dir_exists(path=path)
    execution_graph = execution_graph.copy()
    for node in execution_graph.nodes:
        if "definition" in execution_graph.nodes[node]:
            del execution_graph.nodes[node]["definition"]
    nx.drawing.nx_pydot.write_dot(execution_graph, path)
