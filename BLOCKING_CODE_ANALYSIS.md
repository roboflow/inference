# WebRTC Worker Blocking Code Analysis

## 🚨 CRITICAL BLOCKING SUSPECTS

### 1. **HIGHEST PRIORITY: Video Frame Decoding in Event Loop**
**Location:** `inference/core/interfaces/webrtc_worker/webrtc.py:107-113`

```python
async def recv(self) -> VideoFrame:
    try:
        frame = next(self._iterator)  # ❌ BLOCKING CALL IN ASYNC METHOD!
        return frame
    except StopIteration:
        self.stop()
        raise MediaStreamError("End of video file")
```

**Problem:** `next(self._iterator)` decodes video frames synchronously, which can take 10-100ms+ per frame, completely blocking the event loop.

**Fix:**
```python
async def recv(self) -> VideoFrame:
    loop = asyncio.get_running_loop()
    try:
        frame = await loop.run_in_executor(None, next, self._iterator)
        return frame
    except StopIteration:
        self.stop()
        raise MediaStreamError("End of video file")
```

---

### 2. **Video File Opening in Constructor**
**Location:** `inference/core/interfaces/webrtc_worker/webrtc.py:99-105`

```python
def __init__(self, filepath: str):
    super().__init__()
    import av
    self._container = av.open(filepath)  # ❌ BLOCKING I/O
    self._stream = self._container.streams.video[0]
    self._iterator = self._container.decode(self._stream)
```

**Problem:** `av.open()` does I/O synchronously (opens file, reads headers, initializes codecs).

**Fix:** Create an async factory method:
```python
@classmethod
async def create(cls, filepath: str):
    loop = asyncio.get_running_loop()
    instance = cls.__new__(cls)
    MediaStreamTrack.__init__(instance)

    import av
    instance._container = await loop.run_in_executor(None, av.open, filepath)
    instance._stream = instance._container.streams.video[0]
    instance._iterator = instance._container.decode(instance._stream)
    return instance
```

---

### 3. **Synchronous File Writes During Upload**
**Location:** `inference/core/interfaces/webrtc_worker/webrtc.py:161-176`

```python
def _write_to_temp_file(self) -> None:
    """Reassemble chunks and write to temp file."""
    import tempfile

    total_size = 0
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".mp4", delete=False) as f:  # ❌ BLOCKING I/O
        for i in range(self._total_chunks):
            chunk_data = self._chunks[i]
            f.write(chunk_data)  # ❌ BLOCKING WRITE
            total_size += len(chunk_data)
        self._temp_file_path = f.name
```

**Problem:** This writes potentially hundreds of MB to disk synchronously, called from the data channel message handler (which runs in event loop).

**Fix:** Use `aiofiles` or `run_in_executor`:
```python
async def _write_to_temp_file_async(self) -> None:
    import aiofiles
    import tempfile

    loop = asyncio.get_running_loop()
    fd, path = await loop.run_in_executor(
        None,
        tempfile.mkstemp,
        ".mp4"
    )

    try:
        async with aiofiles.open(fd, mode='wb') as f:
            for i in range(self._total_chunks):
                chunk_data = self._chunks[i]
                await f.write(chunk_data)
        self._temp_file_path = path
        logger.info(f"Video upload complete -> {path}")
        self._chunks.clear()
    except Exception as e:
        await loop.run_in_executor(None, os.unlink, path)
        raise
```

---

### 4. **JSON Serialization in Event Loop**
**Location:** `inference/core/interfaces/webrtc_worker/webrtc.py:440, 460`

```python
# Line 440 - NOT using executor
json_bytes = json.dumps(webrtc_output.model_dump(mode="json")).encode("utf-8")
await send_chunked_data(...)

# Line 460 - NOT using executor
json_bytes = json.dumps(completion_output.model_dump()).encode("utf-8")
await send_chunked_data(...)
```

**Problem:** Large JSON objects (especially with base64-encoded images) can take 10-100ms to serialize.

**Good Example (already fixed at line 397):**
```python
json_bytes = await asyncio.to_thread(
    lambda: json.dumps(webrtc_output.model_dump()).encode("utf-8")
)
```

**Fix:** Apply same pattern to lines 440 and 460.

---

### 5. **Wildcard Serialization (CPU-Bound)**
**Location:** `inference/core/interfaces/webrtc_worker/webrtc.py:429`

```python
try:
    serialized_value = serialize_wildcard_kind(output_data)  # ❌ POTENTIALLY SLOW
    serialized_outputs[field_name] = serialized_value
except Exception as e:
    ...
```

**Problem:** `serialize_wildcard_kind` can encode images to base64, which is CPU-intensive.

**Fix:**
```python
try:
    loop = asyncio.get_running_loop()
    serialized_value = await loop.run_in_executor(
        None,
        serialize_wildcard_kind,
        output_data
    )
    serialized_outputs[field_name] = serialized_value
except Exception as e:
    ...
```

---

### 6. **File Deletion in Event Loop**
**Location:** `inference/core/interfaces/webrtc_worker/webrtc.py:195`

```python
def cleanup(self) -> None:
    """Clean up temp file."""
    if self._temp_file_path:
        import os
        try:
            os.unlink(self._temp_file_path)  # ❌ BLOCKING I/O
        except Exception:
            pass
```

**Fix:** Make it async or schedule on executor:
```python
async def cleanup(self) -> None:
    if self._temp_file_path:
        import os
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, os.unlink, self._temp_file_path)
        except Exception:
            pass
        self._temp_file_path = None
```

---

## ✅ ALREADY CORRECTLY HANDLED

### 1. **Frame Processing** (utils.py:41-108)
The `process_frame()` function (which includes `frame.to_ndarray()`, `VideoFrame.from_ndarray()`, `cv.putText()`) is **correctly offloaded** via:

```python
# webrtc.py:597-607
async def _process_frame_async(self, ...):
    """Async wrapper for process_frame using executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        process_frame,
        frame,
        frame_id,
        self._inference_pipeline,
        ...
    )
```

### 2. **One JSON Operation** (webrtc.py:396-398)
```python
json_bytes = await asyncio.to_thread(
    lambda: json.dumps(webrtc_output.model_dump()).encode("utf-8")
)
```
✅ This one is correctly using `asyncio.to_thread`.

---

## 🔍 INSTRUMENTATION SUGGESTIONS

### Option 1: Asyncio Debug Mode
Add to the top of `webrtc.py`:

```python
import asyncio
import logging

# Enable debug mode
asyncio.get_event_loop().set_debug(True)
asyncio.get_event_loop().slow_callback_duration = 0.1  # Warn if callback takes >100ms

# Or via environment variable: PYTHONASYNCIODEBUG=1
```

### Option 2: Event Loop Watchdog
Add this helper class:

```python
async def event_loop_watchdog(threshold: float = 0.5):
    """Detect when the event loop is blocked."""
    import time
    while True:
        start = time.monotonic()
        await asyncio.sleep(0.1)  # Should take ~100ms
        elapsed = time.monotonic() - start

        if elapsed > threshold:
            logger.warning(f"⚠️ Event loop was blocked for {elapsed:.2f}s")
            # Dump all tasks
            for task in asyncio.all_tasks():
                logger.warning(f"  Task: {task.get_name()} - {task.get_coro()}")
```

Start it in `init_rtc_peer_connection_with_loop()`:
```python
asyncio.create_task(event_loop_watchdog(), name="watchdog")
```

### Option 3: Decorator for Blocking Detection
```python
import functools
import time

def warn_if_blocking(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = func(*args, **kwargs)
        elapsed = time.monotonic() - start
        if elapsed > 0.1:
            logger.warning(f"🚨 {func.__name__} blocked for {elapsed:.2f}s")
            import traceback
            traceback.print_stack()
        return result
    return wrapper

# Apply to suspects:
@warn_if_blocking
def _write_to_temp_file(self):
    ...
```

### Option 4: FaultHandler
Add at module level:

```python
import faulthandler
import signal

# Dump traceback on SIGUSR1
faulthandler.register(signal.SIGUSR1)

# Or dump after timeout (every 30s)
faulthandler.dump_traceback_later(30, repeat=True)
```

Then when hanging: `kill -USR1 <pid>`

---

## 📊 PRIORITY FIX ORDER

1. **CRITICAL:** Fix `OnDemandVideoTrack.recv()` - frame decoding (webrtc.py:109)
2. **HIGH:** Fix `_write_to_temp_file()` - file writes (webrtc.py:166-170)
3. **HIGH:** Fix remaining `json.dumps()` calls (webrtc.py:440, 460)
4. **MEDIUM:** Fix `serialize_wildcard_kind()` (webrtc.py:429)
5. **MEDIUM:** Fix `av.open()` in `OnDemandVideoTrack.__init__()` (webrtc.py:103)
6. **LOW:** Fix `os.unlink()` in cleanup (webrtc.py:195)

---

## 🧪 TESTING APPROACH

1. Enable asyncio debug mode: `PYTHONASYNCIODEBUG=1`
2. Add event loop watchdog (Option 2)
3. Run with video file processing (non-realtime mode)
4. Monitor logs for blocking warnings
5. Use `py-spy` for profiling:
   ```bash
   py-spy top --pid <pid>
   py-spy record --pid <pid> --output profile.svg
   ```

---

## 📝 NOTES

- The `process_frame()` function is **correctly handled** via executor
- The main workflow processing (`inference_pipeline._on_video_frame()`) is also in executor
- The **critical path** is the video decoding in `OnDemandVideoTrack.recv()` which runs in the event loop
- File I/O operations should use `aiofiles` or `run_in_executor`
- CPU-bound operations (JSON serialization, base64 encoding) should use `asyncio.to_thread` or `run_in_executor`
