import os
import time
import sys
from typing import Any, Sequence, Callable, Optional
from loguru import logger
from multiprocessing import Process, RLock, Queue
import numpy as np
import math
from tqdm import tqdm

from lidra.utils.io import touch, stdout_redirected, stderr_redirected
from lidra.test.util import temporary_directory


class Processor:
    STATES = {"done", "failed", "processing"}

    def __init__(
        self,
        path,
        data: Sequence[Any],
        process_fn: Callable,
        timeout: int = 60,  # in minutes
        progress_bar: Optional[bool] = None,
    ):
        if not os.path.exists(path):
            logger.error(f"processor {path} does not exist, creating it ...")
            os.makedirs(path, exist_ok=True)

        self.path = path
        self.timeout = timeout
        self.progress_bar = (
            progress_bar if (progress_bar is not None) else sys.stdout.isatty()
        )
        self._data = data
        self._process_fn = process_fn

    def status(self):
        n = len(self._data)
        status = {
            "total": 0,
            "done": 0,
            "failed": 0,
            "processing": 0,
            "unprocessed": 0,
        }

        for idx in range(n):
            state = self._get_state(idx)
            status["total"] += 1
            status[state] += 1

        return status

    def current_state(self):
        n = len(self._data)
        state = {
            "done": [],
            "failed": [],
            "processing": [],
            "unprocessed": [],
        }

        for idx in range(n):
            state_name = self._get_state(idx)
            state[state_name].append(idx)

        return state

    def done(self):
        return self.current_state()["done"]

    def failed(self):
        return self.current_state()["failed"]

    def processing(self):
        return self.current_state()["processing"]

    def unprocessed(self):
        return self.current_state()["unprocessed"]

    def process(self, n_workers: int = 1):
        tqdm.set_lock(RLock())
        queues_in = [Queue() for _ in range(n_workers)]
        queues_out = [Queue() for _ in range(n_workers)]
        processes = [
            Process(
                target=self._worker,
                args=(
                    worker_idx,
                    tqdm.get_lock(),
                    queues_in[worker_idx],
                    queues_out[worker_idx],
                ),
            )
            for worker_idx in range(n_workers)
        ]

        # start workers
        for process in processes:
            process.start()

        # wait for workers
        last_lines = []
        for queue_in, queue_out, process in zip(queues_in, queues_out, processes):
            queue_in.put(None)  # signal worker to finish
            last_lines.append(queue_out.get())
            process.join()

        if self.progress_bar:
            print("\r", end="")
            for line in last_lines:
                print(line)

    @staticmethod
    def _get_indices(n):
        seed: int = hash((os.getpid(), time.time()))
        seed = seed % (2**32)  # ensure seed is a valid 32-bit integer
        np.random.seed(seed)
        idx = np.arange(n)
        np.random.shuffle(idx)
        return idx

    def _get_idx_state_path(self, idx):
        n = len(self._data)
        name_length = int(math.log10(n)) + 1
        idx_path = os.path.join(self.path, "states", str(idx).zfill(name_length))
        os.makedirs(idx_path, exist_ok=True)
        return idx_path

    def _set_state(self, idx: int, state_to_set: str):
        if not state_to_set in Processor.STATES:
            raise ValueError(
                f"Invalid state: {state_to_set}. Must be one of {Processor.STATES}"
            )
        idx_path = self._get_idx_state_path(idx)

        # touch state to set, remove others
        for state in Processor.STATES:
            state_path = os.path.join(idx_path, state)
            if state == state_to_set:
                touch(state_path)
            else:
                if os.path.exists(state_path):
                    os.remove(state_path)

    def _get_state(self, idx: int):
        idx_path = self._get_idx_state_path(idx)
        active_states = [
            state
            for state in Processor.STATES
            if os.path.exists(os.path.join(idx_path, state))
        ]
        if len(active_states) == 0:
            return "unprocessed"
        elif len(active_states) == 1:
            return active_states[0]
        logger.warning(f"multiple states found for index {idx}: {active_states}")
        most_recent_state = max(
            active_states,
            key=lambda state: os.path.getmtime(os.path.join(idx_path, state)),
        )
        return most_recent_state

    def _is_timed_out(self, idx: int):
        idx_path = self._get_idx_state_path(idx)
        last_modified = os.path.getmtime(idx_path)
        time_now = time.time()
        delta = time_now - last_modified
        return delta > self.timeout * 60

    def _worker(self, worker_idx, tqdm_lock, queue_in: Queue, queue_out: Queue):
        tqdm.set_lock(tqdm_lock)

        indices = Processor._get_indices(len(self._data))
        with tqdm(
            total=len(indices),
            unit="item",
            position=worker_idx,
            desc=f"worker #{worker_idx}",
            dynamic_ncols=True,
            disable=not self.progress_bar,
            lock_args=None,
            leave=True,
        ) as pbar:
            pbar.close = lambda: None  # disable close to avoid tqdm bug
            status = {"skipped": 0, "done": 0, "failed": 0}
            for idx in indices:
                state = self._get_state(idx)

                # already processed
                if state in {"done", "failed"}:
                    status["skipped"] += 1
                # currently being processed
                elif state == "processing" and (not self._is_timed_out(idx)):
                    status["skipped"] += 1
                else:
                    # process data
                    self._set_state(idx, "processing")
                    try:
                        self._process_item(idx, self._data[idx])
                    except:
                        self._set_state(idx, "failed")
                        status["failed"] += 1
                    else:
                        self._set_state(idx, "done")
                        status["done"] += 1

                pbar.set_postfix(status, refresh=False)
                pbar.update()
                pbar.refresh()

            queue_in.get()  # wait for authorization to close (solve tqdm bug)
            queue_out.put(str(pbar))  # send last line to print

    def _process_item(self, idx, item):
        idx_path = self._get_idx_state_path(idx)
        with stdout_redirected(
            os.path.join(idx_path, "stdout.log"),
            file_descriptor_mode=True,
        ), stderr_redirected(
            os.path.join(idx_path, "stderr.log"),
            file_descriptor_mode=True,
        ):
            try:
                self._process_fn(item)
            except:
                logger.opt(exception=True).error(
                    f"error processing item {idx} with data {item}"
                )
                raise
