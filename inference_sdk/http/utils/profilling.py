import json
import os
from datetime import datetime
from typing import List


def save_workflows_profiler_track(
    directory: str,
    track: List[dict],
) -> None:
    os.makedirs(directory, exist_ok=True)
    formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    track_path = os.path.join(
        directory, f"workflow_execution_tack_{formatted_time}.json"
    )
    with open(track_path, "w") as f:
        json.dump(track, f)
