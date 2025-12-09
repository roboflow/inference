# WebRTC SDK Video Upload Implementation Checklist

## Phase 1: Core Upload Infrastructure

### 1.1 Constants and Configuration
- [ ] Add `VIDEO_UPLOAD_CHUNK_SIZE = 48 * 1024` to `inference_sdk/webrtc/config.py`
- [ ] Add `VIDEO_UPLOAD_BUFFER_LIMIT = 256 * 1024` to config
- [ ] Add `VIDEO_UPLOAD_KEEPALIVE_INTERVAL = 5.0` to config

### 1.2 Datachannel Upload Utilities (`inference_sdk/webrtc/datachannel.py`)
- [ ] Add `create_video_upload_chunk(chunk_index, total_chunks, data) -> bytes`
  - Format: `[chunk_index:u32][total_chunks:u32][payload]`
- [ ] Add `VideoFileUploader` class with:
  - [ ] `__init__(self, path: str, channel: RTCDataChannel)`
  - [ ] `async upload(self, on_progress: Optional[Callable] = None)`
  - [ ] Backpressure handling via `bufferedAmount`
  - [ ] Progress callback support

## Phase 2: VideoFileSource Refactoring

### 2.1 Modify `VideoFileSource` (`inference_sdk/webrtc/sources.py`)
- [ ] Remove `_VideoFileTrack` usage from `VideoFileSource`
- [ ] Add `_upload_channel: Optional[RTCDataChannel]` instance variable
- [ ] Add `_uploader: Optional[VideoFileUploader]` instance variable
- [ ] Add `_file_size: int` and `_total_chunks: int` properties

### 2.2 Implement `configure_peer_connection()` for Upload
- [ ] Create `video_upload` datachannel via `pc.createDataChannel("video_upload")`
- [ ] Add `recvonly` video transceiver: `pc.addTransceiver("video", direction="recvonly")`
- [ ] Calculate total chunks from file size
- [ ] Store channel reference for upload

### 2.3 Implement Upload Method
- [ ] Add `async start_upload(self) -> None` method
- [ ] Read file in chunks and send via datachannel
- [ ] Wait for buffer drain between chunks (backpressure)
- [ ] Log progress at intervals

### 2.4 Update `get_initialization_params()`
- [ ] Return `{"video_source": "upload"}` to indicate upload mode
- [ ] Include file metadata if needed by server

## Phase 3: Session Integration

### 3.1 Modify `WebRTCSession._init()` (`inference_sdk/webrtc/session.py`)
- [ ] After `pc.setRemoteDescription(answer)`:
  - [ ] Check if source is `VideoFileSource`
  - [ ] If so, wait for `video_upload` channel to open
  - [ ] Start upload via `asyncio.ensure_future(self._source.start_upload())`

### 3.2 Keepalive Handling
- [ ] Add `_keepalive_task: Optional[asyncio.Task]` to `VideoFileSource`
- [ ] Implement `_keepalive_loop()` sending 1-byte pings every 5s
- [ ] Start keepalive during upload, stop when complete
- [ ] Handle keepalive echo from server (for debugging/metrics)

### 3.3 Channel State Management
- [ ] Handle `video_upload` channel `open` event to start upload
- [ ] Handle `close` event to abort upload gracefully
- [ ] Clean up upload state in `VideoFileSource.cleanup()`

## Phase 4: Error Handling & Edge Cases

### 4.1 Upload Error Handling
- [ ] Handle channel closed during upload → raise appropriate exception
- [ ] Handle file read errors → cleanup and raise
- [ ] Handle timeout → configurable timeout with meaningful error

### 4.2 Cleanup
- [ ] Ensure `VideoFileSource.cleanup()` cancels any pending upload
- [ ] Cancel keepalive task on cleanup
- [ ] Close upload channel if still open

## Phase 5: Testing

### 5.1 Unit Tests (`tests/inference_sdk/unit_tests/webrtc/`)
- [ ] `test_video_upload_chunking.py`:
  - [ ] Test `create_video_upload_chunk()` header format
  - [ ] Test chunk size calculations
  - [ ] Test edge cases (empty file, single chunk, many chunks)

### 5.2 Integration Tests (`tests/inference_sdk/integration_tests/webrtc/`)
- [ ] Update `test_video_file_integration.py`:
  - [ ] Test small video file upload (~1MB)
  - [ ] Verify all chunks received by mock server
  - [ ] Test progress callback invocation

### 5.3 E2E Tests (`tests/inference_sdk/e2e_tests/webrtc/`)
- [ ] Update `test_video_file_e2e.py`:
  - [ ] Test full workflow with actual server
  - [ ] Verify results received via datachannel
  - [ ] Test with video + stream output
  - [ ] Test with data-only output

## Phase 6: Documentation & Cleanup

### 6.1 Code Documentation
- [ ] Add docstrings to new methods
- [ ] Update `VideoFileSource` class docstring to mention upload behavior

### 6.2 Example Update
- [ ] Verify `examples/webrtc_sdk/video_file_basic.py` still works
- [ ] Add optional progress printing to example (demonstration)

### 6.3 Cleanup Legacy Code
- [ ] Remove `_VideoFileTrack` class if no longer needed
- [ ] Remove any obsolete imports

## Implementation Order

1. **Start with**: Phase 1 (config + utilities)
2. **Then**: Phase 2.1-2.3 (VideoFileSource core changes)
3. **Then**: Phase 3.1 (session integration)
4. **Test manually**: Upload should work at this point
5. **Then**: Phase 3.2-3.3 (keepalive + state management)
6. **Then**: Phase 4 (error handling)
7. **Finally**: Phase 5-6 (tests + docs)

## Verification Criteria

- [ ] `examples/webrtc_sdk/video_file_basic.py` works unchanged
- [ ] Video file is chunked and sent via datachannel
- [ ] Server receives all chunks and processes video
- [ ] Results are received via `on_data` handler
- [ ] Video output (if configured) is received via video track
- [ ] No memory leaks with large files
- [ ] Graceful handling of network issues
