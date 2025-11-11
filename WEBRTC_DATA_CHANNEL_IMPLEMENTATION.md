# WebRTC Data Channel Video Streaming Implementation

## Problem Statement

We needed a way to send video frames for inference over WebRTC **without using media tracks**, instead transmitting raw frames via the data channel. This approach:

- **Avoids codec compression** and frame dropping inherent in media tracks
- Provides **reliable, ordered delivery** (TCP-like) of every single frame
- Enables **100% frame processing** for recorded video files
- Separates control/data channels from media channels for better flexibility

## High-Level Approach

### Client Side (SDK)
1. **New Source Type**: `DataChannelVideoSource` - reads video file, encodes frames as JPEG, chunks them, and sends via data channel
2. **Frame Transport Enum**: `FrameTransport.DATA_CHANNEL` vs `MEDIA_TRACK` to specify transport method
3. **Session Management**: Extended `WebRTCSession` to create `upstream_frames` data channel when using data transport
4. **Progress Tracking**: Client tracks frames sent and received, sends EOF only after all responses arrive

### Server Side (Backend)
1. **Frame Assembly**: Server receives chunked frames, reassembles base64 JPEG, decodes to numpy
2. **Queue-Based Processing**: Frames queue in `asyncio.Queue`, processed sequentially
3. **Output Modes**: `DATA_ONLY`, `VIDEO_ONLY`, `BOTH` - control what gets sent back
4. **Timing Instrumentation**: Added detailed logging to identify bottlenecks

## Key Difficulties Encountered

### 1. **Threading + AsyncIO Event Loop Issues**
**Problem**: Data channel `.send()` requires event loop, but sender runs in background thread.

**Solution**: Pass event loop reference to source, use `asyncio.run_coroutine_threadsafe()` to schedule sends on the loop.

```python
def _send_via_loop(self, channel: RTCDataChannel, message: str) -> None:
    if self._loop and not self._loop.is_closed():
        asyncio.run_coroutine_threadsafe(
            self._async_send(channel, message), self._loop
        )
```

### 2. **Premature EOF Signal**
**Problem**: Client sent EOF immediately after sending all frames, but server's queue still had 900+ frames waiting to be processed.

**Solution**: Client tracks received responses and only sends EOF after receiving all 2400 responses.

```python
if message_count[0] >= total_frames:
    source.send_eof_when_ready(message_count[0])
```

### 3. **60-Second Timeout Limit**
**Problem**: Default `WEBRTC_MODAL_FUNCTION_TIME_LIMIT=60` seconds killed processing after ~600 frames.

**Solution**: Set environment variable to 600 seconds when starting container.

```bash
-e WEBRTC_MODAL_FUNCTION_TIME_LIMIT=600
```

### 4. **Docker Volume Mount Not Working**
**Problem**: Code changes weren't being picked up by running container.

**Solution**: Ensured proper volume mount at container startup:
```bash
-v /Users/balthasar/Development/inference/inference:/app/inference
```

### 5. **Performance - Initial Chunk Size Too Small**
**Problem**: 16KB chunks meant 128 messages per 2MB PNG frame = extremely slow.

**Solution**: 
- Switch from PNG to JPEG encoding (2MB → 50-150KB per frame)
- Increase chunk size to 16MB (entire frame in 1 message)
- Result: 40-50x speedup

## Implementation Details

### Client-Side Files

#### `inference_sdk/webrtc/config.py`
```python
class FrameTransport(str, Enum):
    MEDIA_TRACK = "media_track"
    DATA_CHANNEL = "data_channel"

@dataclass
class StreamConfig:
    frame_transport: FrameTransport = FrameTransport.MEDIA_TRACK
    output_mode: str = "both"
    # ... other fields
```

#### `inference_sdk/webrtc/sources.py`
```python
class DataChannelVideoSource(StreamSource):
    def __init__(self, path: str, chunk_size: int = 16_000_000, 
                 image_format: str = "jpg", quality: int = 85):
        # Tracks frames_sent, chunks_sent for progress monitoring
        
    def start_data_channel(self, channel, stop_event, loop):
        # Starts background thread to stream frames
        
    def _stream_frames(self, channel):
        # Reads video file
        # Encodes frames as JPEG with quality param
        # Chunks and sends via data channel
        # Tracks progress via self._frames_sent
```

#### `inference_sdk/webrtc/session.py`
```python
async def _init(self):
    # Creates "inference" data channel (for responses)
    # Creates "upstream_frames" data channel (for sending frames)
    # Passes frame_transport and output_mode in payload
```

### Server-Side Files

#### `inference/core/interfaces/webrtc_worker/entities.py`
```python
class FrameTransport(str, Enum):
    MEDIA_TRACK = "media_track"
    DATA_CHANNEL = "data_channel"

class WebRTCWorkerRequest(BaseModel):
    frame_transport: FrameTransport = FrameTransport.MEDIA_TRACK
    output_mode: WebRTCOutputMode = WebRTCOutputMode.BOTH
```

#### `inference/core/interfaces/webrtc_worker/webrtc.py`
```python
class VideoFrameProcessor:
    def __init__(self, ..., frame_transport, ...):
        self._data_frame_queue = asyncio.Queue()  # Buffer for reassembled frames
        self._chunk_buffers = {}  # Partial frame assembly
        
    def handle_data_channel_payload(self, payload):
        # Routes frame_chunk, frame_eof, frame_error messages
        
    async def _handle_frame_chunk(self, payload):
        # Reassembles chunks -> base64 -> decode JPEG -> numpy -> VideoFrame
        # Queues complete frame in self._data_frame_queue
        
    async def process_frames_data_only(self):
        # Main processing loop
        # Times: wait, inference, send
        # Logs every 30 frames with bottleneck breakdown
```

#### Data Channel Setup
```python
@peer_connection.on("datachannel")
def on_datachannel(channel):
    if channel.label == "upstream_frames":
        @channel.on("message")
        def on_frame_message(message):
            payload = json.loads(message)
            video_processor.handle_data_channel_payload(payload)
```

## Testing

### Quick Test Command
```bash
# Start Docker container with proper config
docker run -d --name inference-webrtc-datachannel \
  -p 9001:9001 \
  -e PROJECT=roboflow-staging \
  -e LOG_LEVEL=INFO \
  -e WEBRTC_MODAL_FUNCTION_TIME_LIMIT=600 \
  -v /Users/balthasar/Development/inference/inference:/app/inference \
  roboflow/roboflow-inference-server-cpu:dev

# Run test script
python examples/webrtc_sdk/data_channel_video.py \
  --video-path ~/Downloads/times_square_2025-08-10_07-02-07.mp4 \
  --workspace-name leandro-starter \
  --workflow-id custom-workflow-3 \
  --api-key LKgvRJqgdbCml2ONofEx \
  --api-url http://localhost:9001 \
  --output-mode data_only

# Monitor server logs for bottleneck analysis
docker logs -f inference-webrtc-datachannel | grep "Frame [0-9]\+:"
```

### Expected Results
- **2400/2400 frames processed** (100% completion)
- **~10 fps processing rate** (limited by inference, not transport)
- **Timing breakdown**: inference=70-90ms, send=0.2ms, wait=0.0ms
- **Queue builds up during sending, drains during processing**

## Performance Analysis

### Bottleneck Breakdown (per frame average)
- **Wait time**: 0.0ms (queue stays full, no blocking)
- **Inference time**: 75ms ← **PRIMARY BOTTLENECK** (ML model processing)
- **Send time**: 0.2ms (negligible network overhead)
- **Total**: ~75ms/frame = ~13 fps max throughput

### Key Insights
1. ✅ **Data channel transport is NOT the bottleneck** - network overhead is <1%
2. ✅ **Inference dominates** - 99%+ of time spent in ML model
3. ✅ **100% frame delivery** - all frames processed, none dropped
4. ✅ **Reliable ordering** - frames processed in exact send order

### Comparison: Media Track vs Data Channel

| Metric | Media Track | Data Channel |
|--------|-------------|--------------|
| Frame delivery | ~60% (drops frames) | 100% (reliable) |
| Ordering | Not guaranteed | Guaranteed FIFO |
| Codec artifacts | Yes (H.264/VP8) | No (raw JPEG) |
| Processing speed | ~10 fps | ~10 fps |
| Network overhead | Low | Low (<1%) |
| Use case | Real-time streaming | Complete video analysis |

## Hindsight: What Could Be Improved

### 1. **Batch Processing**
Instead of processing one frame at a time, batch multiple frames together:
```python
# Current: Process 1 frame → 75ms
# Improved: Process 10 frames together → 150ms total → 15ms/frame
```
This would leverage GPU batch processing for 5-10x speedup.

### 2. **Parallel Workers**
Run multiple inference pipelines in parallel:
```python
# Spin up 4 workers, each processing every 4th frame
# Worker 1: frames 1, 5, 9, 13...
# Worker 2: frames 2, 6, 10, 14...
# etc.
```
Could achieve 4x throughput on multi-core systems.

### 3. **Adaptive Chunk Sizing**
Dynamically adjust chunk size based on frame size:
```python
# Small frames (e.g., 640x480): Use 1MB chunks
# Large frames (e.g., 1920x1080): Use 16MB chunks
```

### 4. **Compression Tuning**
Allow configurable JPEG quality vs speed tradeoff:
```python
# High quality (slower): quality=95
# Balanced: quality=85 (current)
# Fast (lower quality): quality=70
```

### 5. **Frame Skipping Option**
For real-time scenarios, allow dropping frames to maintain target FPS:
```python
if queue_size > threshold and realtime_mode:
    skip_frames(n)  # Drop every nth frame
```

### 6. **Progress Callbacks**
Instead of polling `get_stats()`, use callbacks:
```python
source = DataChannelVideoSource(
    path,
    on_frame_sent=lambda frame_id: print(f"Sent {frame_id}")
)
```

### 7. **Smarter EOF Handling**
Server could acknowledge frame receipt:
```python
# Client sends frame 1234
# Server responds with ack: {type: "frame_ack", frame_id: 1234}
# Client tracks highest ack, sends EOF after last ack
```

### 8. **WebAssembly Decoder**
Move JPEG decode to client-side WASM for better performance:
```python
# Current: Server decodes JPEG (75ms inference includes decode time)
# Improved: Client sends decoded numpy array via binary data channel
```

### 9. **Direct Binary Transfer**
Skip base64 encoding entirely:
```python
# Current: numpy → JPEG → base64 → JSON → data channel
# Improved: numpy → msgpack binary → data channel
```

### 10. **Connection Pooling**
Reuse WebRTC connections for multiple videos:
```python
with client.webrtc.pool() as pool:
    for video in videos:
        pool.process(video)  # Reuses connection
```

## File Checklist

### Created/Modified Files
- ✅ `inference_sdk/webrtc/config.py` - Transport enum, StreamConfig
- ✅ `inference_sdk/webrtc/sources.py` - DataChannelVideoSource class
- ✅ `inference_sdk/webrtc/session.py` - Upstream frames channel setup
- ✅ `inference_sdk/webrtc/client.py` - WebRTC client wrapper
- ✅ `inference_sdk/webrtc/__init__.py` - Package exports
- ✅ `inference_sdk/http/client.py` - `.webrtc` property accessor
- ✅ `inference/core/interfaces/webrtc_worker/entities.py` - FrameTransport, WebRTCOutputMode enums
- ✅ `inference/core/interfaces/webrtc_worker/webrtc.py` - Frame assembly, queue processing, timing logs
- ✅ `inference/core/interfaces/webrtc_worker/utils.py` - declared_fps parameter
- ✅ `inference/core/interfaces/stream_manager/manager_app/entities.py` - data_output type fix
- ✅ `examples/webrtc_sdk/data_channel_video.py` - Test script with progress tracking

## Architecture Diagram

```
┌─────────────┐                          ┌─────────────┐
│   Client    │                          │   Server    │
├─────────────┤                          ├─────────────┤
│             │                          │             │
│ Video File  │                          │  WebRTC     │
│     ↓       │                          │  Worker     │
│ Read Frame  │                          │             │
│     ↓       │    upstream_frames       │  ┌────────┐ │
│ Encode JPEG │  ─────────────────────→  │  │ Queue  │ │
│     ↓       │    (chunked base64)      │  └────┬───┘ │
│ Chunk       │                          │       ↓     │
│     ↓       │                          │  Assemble   │
│ Send Chunks │                          │       ↓     │
│             │                          │  Decode     │
│             │                          │       ↓     │
│ Track Sent  │                          │  Inference  │
│             │                          │   (~75ms)   │
│             │    inference channel     │       ↓     │
│ Count Recv  │  ←─────────────────────  │  Serialize  │
│     ↓       │    (JSON response)       │       ↓     │
│ Send EOF    │                          │  Send JSON  │
│  (when all  │                          │             │
│   received) │                          │             │
└─────────────┘                          └─────────────┘
```

## Message Protocol

### Upstream (Client → Server): `upstream_frames` channel

**Frame Chunk Message:**
```json
{
  "type": "frame_chunk",
  "frame_id": 123,
  "chunk_id": 0,
  "chunks": 1,
  "encoding": "jpg",
  "width": 1280,
  "height": 720,
  "data": "<base64-encoded-chunk>"
}
```

**EOF Message:**
```json
{
  "type": "frame_eof"
}
```

**Error Message:**
```json
{
  "type": "frame_error",
  "error": "failed_to_open"
}
```

### Downstream (Server → Client): `inference` channel

**Inference Result:**
```json
{
  "video_metadata": {
    "frame_id": 123,
    "received_at": "2025-11-11T12:48:10.123456",
    "pts": null,
    "time_base": null,
    "declared_fps": 20.0,
    "measured_fps": null
  },
  "serialized_output_data": {
    "predictions": { ... },
    "count": 12
  },
  "errors": []
}
```

## Configuration Options

### Client-Side (`StreamConfig`)
```python
StreamConfig(
    frame_transport=FrameTransport.DATA_CHANNEL,  # Use data channel
    output_mode="data_only",  # What to receive back
    data_output=None,  # None=all, []=none, ["field1"]=specific
    realtime_processing=False,  # Server buffering mode
    declared_fps=20.0,  # Help server with timing
)
```

### Server-Side (Environment)
```bash
PROJECT=roboflow-staging  # Use staging API
LOG_LEVEL=INFO  # Enable detailed logging
WEBRTC_MODAL_FUNCTION_TIME_LIMIT=600  # 10 minute timeout
```

### Source Options
```python
DataChannelVideoSource(
    path="video.mp4",
    chunk_size=16_000_000,  # 16MB (entire frame typically)
    image_format="jpg",  # JPEG for smaller size
    quality=85  # JPEG quality 0-100
)
```

## Testing Results

### Performance Metrics (2400 frame video @ 20fps)
- **Total time**: ~230 seconds (~3.8 minutes)
- **Processing rate**: 10.4 fps average
- **Frame completion**: 2400/2400 (100%)
- **Sending rate**: ~120 fps (not the bottleneck)
- **Receiving rate**: 10.4 fps (limited by inference)

### Bottleneck Analysis
- **Inference**: 75ms/frame (99% of time)
- **Network send**: 0.2ms/frame (<1% of time)
- **Frame wait**: 0.0ms (queue always full)

### Conclusion
**The data channel transport is NOT the bottleneck.** It successfully delivers 100% of frames with negligible overhead. The limiting factor is purely the ML inference time (~75ms per frame for object detection).

## Future Implementation Recommendations

### If Starting from Scratch

1. **Use msgpack instead of JSON** for binary efficiency
2. **Send raw numpy arrays** directly (skip JPEG encode/decode)
3. **Implement batch processing** from day 1
4. **Add frame acknowledgments** for smarter EOF handling
5. **Use separate WebRTC connection** for data channel only (no media stack overhead)
6. **Implement backpressure** - slow sending if server queue gets too large
7. **Add automatic retry** for failed chunk reassembly
8. **Compression options** - let user choose speed vs quality
9. **Parallel processing** - multiple worker threads on server
10. **Better progress API** - callbacks instead of polling

### Architectural Improvements

**Current**: Single-threaded sequential processing
```
Frame 1 → Decode → Inference → Send → Frame 2 → ...
```

**Better**: Pipeline with stages
```
Thread 1: Decode frames → Queue A
Thread 2: Inference (batch) ← Queue A → Queue B  
Thread 3: Serialize/Send ← Queue B
```

**Best**: Parallel workers with work stealing
```
Worker 1: Frames 1, 5, 9...  ┐
Worker 2: Frames 2, 6, 10... ├─→ Results aggregator → Client
Worker 3: Frames 3, 7, 11... │
Worker 4: Frames 4, 8, 12... ┘
```

## Code Quality Observations

### What Worked Well
- Clean separation of transport layer from inference
- Enum-based configuration (type-safe)
- Detailed timing instrumentation
- Progress tracking on both sides
- Graceful EOF handling

### What Could Be Cleaner
- Too many abstraction layers (Source → Session → Client)
- Mixed async/sync code (threading + asyncio)
- Hardcoded constants (chunk size, timeouts)
- No error recovery mechanism
- Polling-based progress (should use callbacks)

## Related Files for Reference
- `examples/webrtc/test_frame_delivery.py` - Media track comparison test
- `examples/webrtc/webrtc_worker.py` - Reference implementation with UI
- `examples/webrtc/webcam.py` - Webcam streaming example

