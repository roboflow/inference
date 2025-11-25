import os
import datetime
import tracemalloc
import psutil
from loguru import logger
import json

MAX_N_FRAMES = 128

tracemalloc.start(MAX_N_FRAMES)


def get_memory_info(pid=None):
    return {
        "system": psutil.virtual_memory()._asdict(),
        "pid": os.getpid() if pid is None else pid,
        "process": get_process_memory_info(pid),
    }


def get_process_memory_info(pid=None):
    pid = os.getpid() if pid is None else pid

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []

    children = process.children(recursive=False)

    result = {
        str(pid): {
            "pid": pid,
            "name": process.name(),
            "memory": {
                "info": process.memory_info()._asdict(),
                "map": [mm._asdict() for mm in process.memory_maps()],
            },
            "parent": process.parent().pid,
            "children": [child.pid for child in children],
        }
    }

    for child in children:
        child_info = get_process_memory_info(child.pid)
        result.update(child_info)

    return result


def dump_memory_info(folder="."):
    # create folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    pid = os.getpid()
    folder = os.path.join(folder, f"{pid}/{timestamp}")
    os.makedirs(folder, exist_ok=False)
    logger.info(f"dump memory info here : {folder}")

    # dump tracemalloc snaphot (python traces)
    snapshop = tracemalloc.take_snapshot()
    snapshop.dump(os.path.join(folder, "snapshot.dat"))

    # dump memory info (memory per process + memory maps)
    memory_info = get_memory_info()
    with open(os.path.join(folder, "mem_info.json"), "w") as f:
        json.dump(memory_info, f, indent=2)

    return folder
