# WebRTC SDK Video File Upload via DataChannel

## Task Overview

**Type**: Complex Feature / Refactoring
**Goal**: Adapt the Python WebRTC SDK to handle video files via datachannel upload instead of streaming frames through WebRTC media tracks, while keeping the public API unchanged.

**Current Behavior**: `VideoFileSource` creates a `_VideoFileTrack` that reads frames from video files using PyAV and streams them as a WebRTC video track.

**Target Behavior**: `VideoFileSource` reads the video file as binary, chunks it, and uploads via a `video_upload` datachannel. The server reassembles the file, processes it, and streams results back.

## Server Protocol Analysis (PR #1778)

The server already supports video file upload via datachannel. Key components:

### 1. Upload Protocol
- **Channel name**: `video_upload`
- **Header format**: `[chunk_index:u32][total_chunks:u32][payload]` (8-byte header)
- **Chunk size**: Recommended ~48KB for safe WebRTC transmission
- **Completion**: Auto-detected when all chunks are received

### 2. Server-Side Handler (`webrtc.py:1083-1118`)
```python
if channel.label == "video_upload":
    video_processor.video_upload_handler = VideoFileUploadHandler()

    @channel.on("message")
    def on_upload_message(message):
        # Keepalive pings (1-byte messages)
        if len(message) <= 1:
            channel.send(message)  # Echo back
            return

        chunk_index, total_chunks, data = parse_video_file_chunk(message)
        video_processor.video_upload_handler.handle_chunk(...)

        # Auto-start processing when complete
        video_path = video_processor.video_upload_handler.try_start_processing()
        if video_path:
            player = MediaPlayer(video_path)
            video_processor.set_track(track=player.video)
```

### 3. State Machine
```
IDLE → UPLOADING → COMPLETE → PROCESSING
```

### 4. Keepalive Mechanism
Server echoes 1-byte keepalive pings back to maintain TURN connection during long uploads.

## Current SDK Architecture

### Sources (`sources.py`)
- `StreamSource` (ABC) - base interface with:
  - `configure_peer_connection(pc)` - configure WebRTC connection
  - `get_initialization_params()` - params for server `/initialise_webrtc_worker`
  - `cleanup()` - release resources
- `VideoFileSource` - current implementation using `_VideoFileTrack`

### Session (`session.py`)
- `WebRTCSession._init()` - initializes peer connection
- Creates `inference` datachannel for bidirectional data
- Processes incoming video frames from track

## Solution Design

### Recommended Approach: Internal Refactoring with Optional Behavior

Keep `VideoFileSource` API identical, but internally use datachannel upload instead of video track streaming.

#### Changes Required

**1. Modify `VideoFileSource.configure_peer_connection()` (`sources.py`)**

Instead of adding a video track, create a `video_upload` datachannel:

```python
async def configure_peer_connection(self, pc: RTCPeerConnection) -> None:
    """Configure peer connection for video file upload via datachannel."""
    # Create video_upload channel (server will create handler when it receives this)
    self._upload_channel = pc.createDataChannel("video_upload")

    # Add recvonly transceiver to receive video back from server
    pc.addTransceiver("video", direction="recvonly")

    # Store file size and calculate chunks
    self._file_size = os.path.getsize(self.path)
    self._total_chunks = (self._file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
```

**2. Add upload method to `VideoFileSource`**

```python
async def upload_file(self, on_progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Upload video file through datachannel."""
    with open(self.path, 'rb') as f:
        for chunk_idx in range(self._total_chunks):
            chunk_data = f.read(CHUNK_SIZE)
            message = struct.pack("<II", chunk_idx, self._total_chunks) + chunk_data

            # Wait for channel buffer to drain (backpressure)
            while self._upload_channel.bufferedAmount > BUFFER_LIMIT:
                await asyncio.sleep(0.01)

            self._upload_channel.send(message)

            if on_progress:
                on_progress(chunk_idx + 1, self._total_chunks)
```

**3. Modify `WebRTCSession._init()` to start upload (`session.py`)**

After peer connection is established, if source is `VideoFileSource`, start the upload:

```python
# After remote description is set
if isinstance(self._source, VideoFileSource):
    asyncio.ensure_future(self._source.upload_file())
```

**4. Add keepalive handling**

Send periodic 1-byte keepalives and handle echoes to maintain TURN connection:

```python
async def _keepalive_loop(self):
    while self._upload_channel.readyState == "open":
        self._upload_channel.send(b'\x00')  # Keepalive ping
        await asyncio.sleep(5)  # Every 5 seconds
```

### Key Implementation Details

1. **Chunk Size**: Use 48KB (`48 * 1024`) for safe WebRTC transmission
2. **Buffer Management**: Monitor `bufferedAmount` to prevent overflow
3. **Progress Callbacks**: Optional progress reporting during upload
4. **Keepalive**: Send 1-byte pings every 5s during upload
5. **Error Handling**: Handle channel close during upload gracefully

### Files to Modify

| File | Changes |
|------|---------|
| `inference_sdk/webrtc/sources.py` | Refactor `VideoFileSource` to use datachannel upload |
| `inference_sdk/webrtc/session.py` | Add upload orchestration after connection setup |
| `inference_sdk/webrtc/datachannel.py` | Add upload-related utilities (chunking, progress) |
| `inference_sdk/webrtc/config.py` | Add upload-related config (chunk size, buffer limits) |

### API Compatibility

The public API remains unchanged:

```python
# This code continues to work exactly as before
source = VideoFileSource(args.video_path)
session = client.webrtc.stream(
    source=source,
    workflow=args.workflow_id,
    workspace=args.workspace_name,
)
session.run()
```

## Testing Strategy

1. **Unit Tests**: Mock datachannel, verify chunking/header format
2. **Integration Tests**: Upload small video file to local server
3. **E2E Tests**: Full workflow with video file processing

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Large files may timeout | Implement progress tracking, keepalives |
| Buffer overflow | Backpressure via `bufferedAmount` check |
| Network interruption | Consider resume capability (future) |
| TURN connection drop | Keepalive mechanism maintains connection |

## Dependencies

- Server must support `video_upload` channel (already in PR #1778)
- No new Python dependencies required
