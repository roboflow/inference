import json
import os
from datetime import datetime
from typing import List


def save_workflows_profiler_trace(
    directory: str,
    profiler_trace: List[dict],
) -> None:
    """Save a workflow profiler trace.

    Args:
        directory: The directory to save the profiler trace.
        profiler_trace: The profiler trace.
    """
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    track_path = os.path.join(
        directory, f"workflow_execution_tack_{formatted_time}.json"
    )
    with open(track_path, "w") as f:
        json.dump(profiler_trace, f)
