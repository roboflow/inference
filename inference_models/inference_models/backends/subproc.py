import asyncio
import io
import json
import multiprocessing as mp
import pickle
import threading
import traceback
from multiprocessing.reduction import ForkingPickler
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List, Optional
from uuid import uuid4

import zmq

from inference_models.backends.utils.shm_serializer import pack, unpack

_DEFAULT_INPUT_SHM_SIZE = 100 * 1024 * 1024   # 100 MB
_WORKER_READY_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# CUDA IPC helpers
# ---------------------------------------------------------------------------

def _has_cuda_input(*args, **kwargs) -> bool:
    """Returns True if any argument is a CUDA tensor."""
    try:
        import torch
        return any(
            isinstance(a, torch.Tensor) and a.is_cuda
            for a in (*args, *kwargs.values())
        )
    except ImportError:
        return False


def _ipc_pickle(obj: Any) -> bytes:
    """Pickle using PyTorch's IPC-aware pickler.

    For CUDA tensors this produces a small IPC handle (not a copy of the GPU
    data). The receiver unpickles normally — PyTorch reconstructs the tensor
    from the handle, giving both processes a view of the same GPU memory.
    """
    import torch.multiprocessing.reductions  # registers CUDA tensor reducers  # noqa: F401
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Worker entry point (module-level — must be picklable for spawn)
# ---------------------------------------------------------------------------

def _worker_main(
    model_id: str,
    api_key: str,
    socket_addr: str,
    input_shm_name: str,
    result_shm_name: str,
    model_kwargs: dict,
) -> None:
    """Entry point for a spawned worker process.

    Loads one model and serves inference requests.

    Transport:
        "shm"      — arrays written to shared memory (zero-copy); descriptors +
                      pickled remainder sent via ZeroMQ.  Input uses input_shm,
                      results use result_shm.
        "cuda_ipc" — IPC-aware pickle via ZeroMQ frame (no GPU→CPU copy).

    Protocol (all ZeroMQ messages are multipart [header_json, payload]):
        manager → worker:  {"type": "infer", "transport": "shm",
                            "descriptors": [...]},                    <pickled_remainder>
                        OR {"type": "infer", "transport": "cuda_ipc"}, <ipc_bytes>
        worker  → manager: {"type": "result", "transport": "shm",
                            "descriptors": [...]},                    <pickled_remainder>
                        OR {"type": "result"},                         <ipc_bytes>
                        OR {"type": "error", "traceback": "..."},      b""
        manager → worker:  {"type": "shutdown"},                       b""
    """
    zmq_ctx = zmq.Context()
    socket = zmq_ctx.socket(zmq.PAIR)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(socket_addr)

    input_shm = SharedMemory(name=input_shm_name)
    result_shm = SharedMemory(name=result_shm_name)

    def send(header: dict, payload: bytes = b"") -> None:
        socket.send_multipart([json.dumps(header).encode(), payload])

    try:
        try:
            from inference_models.models.auto_loaders.core import AutoModel
            model = AutoModel.from_pretrained(model_id, api_key=api_key, **model_kwargs)
            print(model.__class__.__name__)
            class_names = getattr(model, "class_names", None)
            send({"status": "ready", "class_names": class_names})
        except Exception as e:
            send({"status": "error", "error": str(e), "traceback": traceback.format_exc()})
            return

        while True:
            header_bytes, payload = socket.recv_multipart()
            msg = json.loads(header_bytes)

            if msg["type"] == "shutdown":
                break

            if msg["type"] == "infer":
                try:
                    if msg["transport"] == "shm":
                        args, kwargs = unpack(
                            msg["descriptors"], payload, input_shm.buf, copy=False,
                        )
                    else:  # cuda_ipc
                        args, kwargs = pickle.loads(payload)

                    result = model.infer(*args, **kwargs)
                    del args, kwargs

                    if msg["transport"] == "shm":
                        descriptors, pickled, _ = pack(result, result_shm.buf)
                        send(
                            {"type": "result", "transport": "shm", "descriptors": descriptors},
                            pickled,
                        )
                    else:
                        send({"type": "result"}, _ipc_pickle(result))
                except Exception as e:
                    send({"type": "error", "traceback": traceback.format_exc()})
                finally:
                    args = kwargs = None  # ensure cleanup even on error path
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
    finally:
        input_shm.close()
        result_shm.close()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        socket.close()
        zmq_ctx.term()

# ---------------------------------------------------------------------------
# SubprocessBackend
# ---------------------------------------------------------------------------

class SubprocessBackend(Backend):
    """Loads a model in a dedicated spawned worker process.

    Data plane:
    - Input (CPU/numpy): written to pre-allocated shared memory (zero-copy on CPU)
    - Input (CUDA tensor): sent as a ZeroMQ frame containing a small CUDA IPC
      handle — no GPU→CPU copy, both processes share the same GPU memory
    - Output: received as a ZeroMQ frame; CUDA tensors in the result are
      handled transparently via CUDA IPC inside the pickle payload

    Control: ZeroMQ PAIR socket (also carries result frames).

    Uses start_method="spawn" — fork is not viable because CUDA cannot be
    used after fork().

    Preferred for: HTTP server, GPU machines, multi-worker deployments.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        input_shm_size: int = _DEFAULT_INPUT_SHM_SIZE,
        worker_ready_timeout_s: int = _WORKER_READY_TIMEOUT_S,
        transport: str = "auto",
        **kwargs,
    ) -> None:
        """Spawns the worker process, passes config, blocks until ready signal.

        Args:
            transport: "auto" — cuda_ipc if CUDA is available, else shm.
                       "shm"  — always use shared memory.
                       "cuda_ipc" — always use CUDA IPC handles via ZeroMQ.
        """
        if transport == "auto":
            try:
                import torch
                transport = "cuda_ipc" if torch.cuda.is_available() else "shm"
            except ImportError:
                transport = "shm"
        elif transport == "cuda_ipc":
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("transport='cuda_ipc' requires CUDA but no GPU is available")
            except ImportError:
                raise RuntimeError("transport='cuda_ipc' requires PyTorch with CUDA support")
        elif transport != "shm":
            raise ValueError(f"Unknown transport {transport!r}, expected 'auto', 'shm', or 'cuda_ipc'")
        self._transport = transport
        self._model_id = model_id

        # ZeroMQ control + result channel — PAIR: point-to-point, bidirectional, ordered.
        # Not thread-safe; _socket_lock must be held for every send/recv pair.
        socket_addr = f"ipc:///tmp/inference_worker_{uuid4().hex}"
        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.PAIR)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s recv timeout — allows shutdown
        self._socket.bind(socket_addr)
        self._shutting_down = False
        self._socket_lock = threading.Lock()
        # Async callers queue here as suspended coroutines — no thread consumed per waiter.
        self._async_lock = asyncio.Lock()

        # Pre-allocated shared memory for CPU/numpy data (both directions).
        self._input_shm = SharedMemory(create=True, size=input_shm_size)
        self._result_shm = SharedMemory(create=True, size=input_shm_size)

        # Spawn worker — not fork, CUDA cannot be used after fork().
        ctx = mp.get_context("spawn")
        self._process = ctx.Process(
            target=_worker_main,
            kwargs=dict(
                model_id=model_id,
                api_key=api_key,
                socket_addr=socket_addr,
                input_shm_name=self._input_shm.name,
                result_shm_name=self._result_shm.name,
                model_kwargs=kwargs,
            ),
            daemon=True,
        )
        self._process.start()

        # Block until the worker signals ready (or fails).
        if not self._socket.poll(timeout=worker_ready_timeout_s * 1000):
            self._cleanup()
            raise RuntimeError(
                f"Worker for model '{model_id}' did not signal ready "
                f"within {worker_ready_timeout_s}s"
            )
        header_bytes, _ = self._socket.recv_multipart()
        msg = json.loads(header_bytes)
        if msg.get("status") != "ready":
            self._cleanup()
            raise RuntimeError(
                f"Worker for model '{model_id}' failed to initialize: "
                f"{msg.get('error', msg)}"
            )
        self._class_names = msg.get("class_names")

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names

    @property
    def is_healthy(self) -> bool:
        # TODO: add background sentinel or periodic heartbeat for proactive detection
        return self._process.is_alive() and not self._shutting_down

    @property
    def max_batch_size(self) -> Optional[int]:
        # TODO: query from worker (model metadata / TRT engine profile)
        return None

    def stats(self) -> Dict[str, Any]:
        # TODO: track inference_count, total_latency_s, last_latency_s, errors
        return {}

    def _cleanup(self) -> None:
        """Terminate process and release all IPC resources."""
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
        self._socket.close()
        self._zmq_context.term()
        self._input_shm.close()
        self._input_shm.unlink()
        self._result_shm.close()
        self._result_shm.unlink()

    def unload(self) -> None:
        """Sends shutdown message to the worker process and cleans up."""
        self._shutting_down = True
        try:
            self._socket.send_multipart(
                [json.dumps({"type": "shutdown"}).encode(), b""],
                flags=zmq.NOBLOCK,
            )
            self._process.join(timeout=3)
        except zmq.ZMQError:
            pass
        self._cleanup()

    async def infer_async(self, *args, **kwargs) -> Any:
        """Serializes input, sends to worker, and awaits result.

        Concurrent callers queue on _async_lock as suspended coroutines — no
        thread pool thread is consumed per waiter. Only the blocking recv_multipart
        call runs on the executor (one thread at a time).

        Transport selection (same as infer_sync):
        - CUDA tensor input → cuda_ipc (ZeroMQ frame, no GPU→CPU copy)
        - CPU / numpy input → shm (shared memory write)
        """
        loop = asyncio.get_running_loop()

        use_cuda_ipc = self._transport == "cuda_ipc" and _has_cuda_input(*args, **kwargs)

        if use_cuda_ipc:
            zmq_payload = _ipc_pickle((args, kwargs))
            header = json.dumps({"type": "infer", "transport": "cuda_ipc"}).encode()
        else:
            descriptors, pickled, _ = pack((args, kwargs), self._input_shm.buf)
            header = json.dumps({
                "type": "infer", "transport": "shm", "descriptors": descriptors,
            }).encode()
            zmq_payload = pickled

        def _recv_with_timeout():
            while not self._shutting_down:
                try:
                    return self._socket.recv_multipart()
                except zmq.Again:
                    continue
            raise RuntimeError("Backend is shutting down")

        async with self._async_lock:
            self._socket.send_multipart([header, zmq_payload])
            header_bytes, result_payload = await loop.run_in_executor(
                None, _recv_with_timeout
            )

        msg = json.loads(header_bytes)
        if msg["type"] == "error":
            raise RuntimeError(
                f"Worker for model '{self._model_id}' raised an exception:\n"
                f"{msg['traceback']}"
            )
        if use_cuda_ipc:
            return pickle.loads(result_payload)
        return unpack(msg["descriptors"], result_payload, self._result_shm.buf, copy=True)

    def infer_sync(self, *args, **kwargs) -> Any:
        """Sends input to worker and blocks for result.

        Transport selection:
        - CUDA tensor input → cuda_ipc: IPC-aware pickle into ZeroMQ frame
        - CPU / numpy input → shm: raw array bytes in shared memory,
          descriptors + pickled remainder via ZeroMQ
        """
        use_cuda_ipc = self._transport == "cuda_ipc" and _has_cuda_input(*args, **kwargs)

        if use_cuda_ipc:
            payload = _ipc_pickle((args, kwargs))
            header = json.dumps({"type": "infer", "transport": "cuda_ipc"}).encode()
            send_frames = [header, payload]
        else:
            descriptors, pickled, _ = pack((args, kwargs), self._input_shm.buf)
            header = json.dumps({
                "type": "infer", "transport": "shm", "descriptors": descriptors,
            }).encode()
            send_frames = [header, pickled]

        with self._socket_lock:
            self._socket.send_multipart(send_frames)
            while not self._shutting_down:
                try:
                    header_bytes, result_payload = self._socket.recv_multipart()
                    break
                except zmq.Again:
                    continue
            else:
                raise RuntimeError("Backend is shutting down")

        msg = json.loads(header_bytes)
        if msg["type"] == "error":
            raise RuntimeError(
                f"Worker for model '{self._model_id}' raised an exception:\n"
                f"{msg['traceback']}"
            )
        if use_cuda_ipc:
            return pickle.loads(result_payload)
        return unpack(msg["descriptors"], result_payload, self._result_shm.buf, copy=True)
