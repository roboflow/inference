"""
Test frame delivery completeness for video files across different modes.

Measures:
1. data_only + realtime: frames via data channel with realtime processing
2. data_only + no-realtime: frames via data channel with buffering
3. both + realtime: frames via both channels with realtime
4. both + no-realtime: frames via both channels with buffering

Usage:
  python examples/webrtc/test_frame_delivery.py \
      --video-path ~/Downloads/times_square_2025-08-10_07-02-07.mp4 \
      --workspace-id leandro-starter \
      --workflow-id custom-workflow-3 \
      --inference-server-url http://localhost:9001 \
      --api-key LKgvRJqgdbCml2ONofEx
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from threading import Thread
from typing import List, Optional

import cv2 as cv
import numpy as np
import requests
from aiortc import RTCDataChannel, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame
from av.logging import ERROR, set_libav_level

from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCOffer,
    WebRTCTURNConfig,
    WorkflowConfiguration,
)
from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest
from inference.core.roboflow_api import get_workflow_specification
from inference.core.utils.async_utils import Queue

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s",
    force=True,
)
logger = logging.getLogger(Path(__file__).stem)

# Debug: confirm imports complete
print("✓ All imports loaded successfully", file=sys.stderr, flush=True)


class FramesGrabber:
    def __init__(self, source_path: str):
        self._cap = cv.VideoCapture(source_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {source_path}")
        self._fps = self._cap.get(cv.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))

    def get_frame(self) -> Optional[np.ndarray]:
        ret, np_frame = self._cap.read()
        if not ret:
            return None
        return np_frame

    def get_fps(self) -> Optional[float]:
        return self._fps
    
    def get_total_frames(self) -> int:
        return self._total_frames


class StreamTrack(VideoStreamTrack):
    def __init__(
        self,
        asyncio_loop: Optional[asyncio.AbstractEventLoop] = None,
        source_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loop = asyncio_loop
        if asyncio_loop is None:
            self._loop = asyncio.get_event_loop()

        self._source: Optional[FramesGrabber] = None
        if source_path is not None:
            self._source = FramesGrabber(source_path=source_path)

        self.track: Optional[RemoteStreamTrack] = None
        self._recv_task: Optional[asyncio.Task] = None
        self.recv_queue: "Queue[Optional[VideoFrame]]" = Queue(loop=self._loop)
        self._av_logging_set: bool = False
        self._frames_sent = 0

    def set_track(self, track: RemoteStreamTrack):
        self.track = track
        self._recv_task = self._loop.create_task(self._recv_loop(), name="recv_loop")

    async def stop_recv_loop(self):
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None
        await self.recv_queue.async_put(None)

    async def _recv_loop(self):
        try:
            while True:
                frame = await self.track.recv()
                await self.recv_queue.async_put(frame)
        except Exception as e:
            print(f"Recv loop ended: {e}")
        finally:
            await self.recv_queue.async_put(None)

    async def recv(self):
        if not self._av_logging_set:
            set_libav_level(ERROR)
            self._av_logging_set = True

        if self._source:
            # Send from local source
            np_frame = self._source.get_frame()
            if np_frame is None:
                print(f"Video file ended after {self._frames_sent} frames", flush=True)
                raise Exception("End of video file")

            self._frames_sent += 1
            vf = VideoFrame.from_ndarray(np_frame, format="bgr24")
            vf.pts, vf.time_base = await self.next_timestamp()
            return vf
        else:
            # Receive from remote
            frame = await self.recv_queue.async_get()
            if frame is None:
                raise Exception("Stream ended")
            return frame


def run_test(
    video_path: str,
    workspace_id: str,
    workflow_id: str,
    api_key: str,
    inference_server_url: str,
    output_mode: str,
    realtime: bool,
    stream_output_param: Optional[str] = None,
    data_output_param: Optional[str] = None,
) -> dict:
    """Run a single test configuration."""
    
    print(f"DEBUG: Entered run_test(), mode={output_mode}, realtime={realtime}", file=sys.stderr, flush=True)
    print("="*80, flush=True)
    print(f"TEST: output_mode={output_mode}, realtime={realtime}", flush=True)
    print("="*80, flush=True)
    
    # Get video metadata
    print(f"DEBUG: About to read video metadata", file=sys.stderr, flush=True)
    print(f"[1/7] Reading video metadata from {video_path}...")
    print(f"DEBUG: About to open cv.VideoCapture", file=sys.stderr, flush=True)
    cap = cv.VideoCapture(video_path)
    print(f"DEBUG: VideoCapture opened", file=sys.stderr, flush=True)
    print(f"DEBUG: Getting frame count...", file=sys.stderr, flush=True)
    total_frames_in_video = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"DEBUG: Got {total_frames_in_video} frames, getting FPS...", file=sys.stderr, flush=True)
    video_fps = cap.get(cv.CAP_PROP_FPS)
    print(f"DEBUG: Got FPS {video_fps}, releasing...", file=sys.stderr, flush=True)
    cap.release()
    print(f"DEBUG: VideoCapture released", file=sys.stderr, flush=True)
    
    print(f"      ✓ Video file: {total_frames_in_video} frames @ {video_fps:.2f} FPS")
    
    # Metrics
    metrics = {
        "output_mode": output_mode,
        "realtime": realtime,
        "total_frames_in_video": total_frames_in_video,
        "video_fps": video_fps,
        "frames_sent": 0,
        "video_frames_received": 0,
        "data_messages_received": 0,
        "unique_frame_ids": set(),
        "start_time": None,
        "end_time": None,
        "errors": [],
        "connection_states": [],
        "data_channel_states": [],
    }
    
    # Get workflow spec
    print(f"[2/7] Fetching workflow specification...")
    workflow_specification = get_workflow_specification(
        api_key=api_key,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
    )
    print(f"      ✓ Workflow loaded")
    
    # Determine outputs
    stream_output = stream_output_param
    data_output_selection: Optional[List[str]] = None

    outputs = workflow_specification.get("outputs", [])

    available_output_names = [o.get("name") for o in outputs]

    if stream_output is not None:
        if stream_output not in available_output_names:
            raise ValueError(
                f"Requested stream_output '{stream_output}' not found in workflow outputs. Available: {available_output_names}"
            )
    elif output_mode in ["both", "video_only"] and outputs:
        stream_output = outputs[0].get("name")

    if data_output_param is not None:
        requested = [part.strip() for part in data_output_param.split(",") if part.strip()]
        if len(requested) == 0 or requested == ["none"]:
            data_output_selection = []
        elif requested == ["all"]:
            data_output_selection = None
        else:
            # Validate requested fields
            invalid = [name for name in requested if name not in available_output_names]
            if invalid:
                raise ValueError(
                    f"Invalid data_output fields {invalid}. Available: {available_output_names}"
                )
            data_output_selection = requested
    else:
        if output_mode in ["both", "data_only"]:
            data_output_selection = None  # None means "all outputs"
        else:
            data_output_selection = []

    print(f"      stream_output: {stream_output}")
    if data_output_selection is None:
        print("      data_output: ALL")
    elif len(data_output_selection) == 0:
        print("      data_output: NONE")
    else:
        print(f"      data_output: {data_output_selection}")
    
    # Setup asyncio loop
    print(f"[3/7] Setting up asyncio event loop...")
    asyncio_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(asyncio_loop)
    
    def run_loop():
        asyncio_loop.run_forever()
    
    thread = Thread(target=run_loop, daemon=True)
    thread.start()
    print(f"      ✓ Event loop running")
    
    # Create peer connection
    print(f"[4/7] Creating WebRTC peer connection...")
    peer_connection = RTCPeerConnection()
    peer_connection.closed_event = asyncio.Event()
    peer_connection.stream_track = StreamTrack(
        asyncio_loop=asyncio_loop,
        source_path=video_path,
    )
    metrics["connection_states"].append(peer_connection.connectionState)
    
    # Add connection state handler
    @peer_connection.on("connectionstatechange")
    def on_connection_state():
        state = peer_connection.connectionState
        metrics["connection_states"].append(state)
        print(f"DEBUG: Connection state changed to: {state}", flush=True)
        if state in {"closed", "failed"}:
            asyncio_loop.call_soon_threadsafe(peer_connection.closed_event.set)
    
    # Add local track
    peer_connection.addTrack(peer_connection.stream_track)
    print(f"      ✓ Local video track added")
    
    # Setup remote track handler
    relay = MediaRelay()
    
    @peer_connection.on("track")
    def on_track(track):
        print("Remote track received")
        relayed_track = relay.subscribe(track, buffered=False)
        peer_connection.stream_track.set_track(relayed_track)
    
    def handle_data_message(message: str) -> None:
        metrics["data_messages_received"] += 1
        if metrics["data_messages_received"] <= 3:
            preview = message if len(message) < 200 else f"{message[:200]}..."
            print(f"DEBUG: Data message #{metrics['data_messages_received']}: {preview}", flush=True)
        try:
            parsed = json.loads(message)
            metadata = parsed.get("video_metadata") or {}
            frame_id = metadata.get("frame_id")
            if frame_id is not None:
                metrics["unique_frame_ids"].add(frame_id)
        except Exception as e:
            metrics["errors"].append(f"Failed to parse data message: {e}")
            if metrics["data_messages_received"] <= 3:
                print(f"DEBUG: Parse error: {e}", flush=True)

    # Client-initiated data channel
    data_channel = peer_connection.createDataChannel("inference")
    peer_connection.data_channel = data_channel
    metrics["data_channel_states"].append("created")

    @data_channel.on("open")
    def _on_data_open():
        metrics["data_channel_states"].append("open")
        print("DEBUG: Data channel opened", flush=True)

    @data_channel.on("close")
    def _on_data_close():
        metrics["data_channel_states"].append("closed")
        print("DEBUG: Data channel closed", flush=True)

    @data_channel.on("message")
    def _on_data_message(message):
        handle_data_message(message)

    # Setup data channel handler for server-initiated channel (fallback)
    @peer_connection.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        print(f"Data channel '{channel.label}' received from server")

        @channel.on("open")
        def _open():
            metrics["data_channel_states"].append("open (server)")
            print("DEBUG: Server data channel opened", flush=True)

        @channel.on("close")
        def _close():
            metrics["data_channel_states"].append("closed (server)")
            print("DEBUG: Server data channel closed", flush=True)

        @channel.on("message")
        def on_message(message):
            handle_data_message(message)

        peer_connection.data_channel = channel
    
    # Create offer
    print(f"[5/7] Negotiating WebRTC connection with server...")
    async def setup():
        offer = await peer_connection.createOffer()
        await peer_connection.setLocalDescription(offer)
        
        # Wait for ICE gathering
        while peer_connection.iceGatheringState != "complete":
            await asyncio.sleep(0.1)
        
        print(f"      ✓ ICE gathering complete, sending offer to server...")
        # Send to server
        request = WebRTCWorkerRequest(
            api_key=api_key,
            workflow_configuration=WorkflowConfiguration(
                type="WorkflowConfiguration",
                workflow_id=workflow_id,
                workspace_name=workspace_id,
                image_input_name="image",
            ),
            webrtc_offer=WebRTCOffer(
                type=peer_connection.localDescription.type,
                sdp=peer_connection.localDescription.sdp,
            ),
            output_mode=output_mode,
            stream_output=[stream_output] if stream_output else [],
            data_output=data_output_selection,
            webrtc_realtime_processing=realtime,
            declared_fps=video_fps if video_fps and video_fps > 0 else None,
        )
        
        response = requests.post(
            f"{inference_server_url}/initialise_webrtc_worker",
            json=request.model_dump(),
            timeout=30,
        )
        response.raise_for_status()
        answer_data = response.json()
        
        answer = RTCSessionDescription(
            sdp=answer_data["sdp"],
            type=answer_data["type"],
        )
        await peer_connection.setRemoteDescription(answer)
        
        print(f"      ✓ WebRTC connection established!")
    
    # Run setup
    asyncio.run_coroutine_threadsafe(setup(), asyncio_loop).result()
    
    metrics["start_time"] = time.time()
    
    # Process frames
    print(f"[6/7] Processing video (total {total_frames_in_video} frames)...")
    last_progress_time = time.time()
    try:
        if output_mode != "data_only":
            # Receive video frames
            while True:
                try:
                    frame = asyncio.run_coroutine_threadsafe(
                        peer_connection.stream_track.recv_queue.async_get(),
                        asyncio_loop,
                    ).result(timeout=5)
                    
                    if frame is None:
                        print("      ✓ Video stream ended")
                        break
                    
                    metrics["video_frames_received"] += 1
                    
                    # Progress update every 2 seconds
                    now = time.time()
                    if now - last_progress_time > 2.0:
                        progress = metrics["video_frames_received"] / total_frames_in_video * 100
                        print(f"      Progress: {metrics['video_frames_received']}/{total_frames_in_video} frames ({progress:.1f}%)")
                        last_progress_time = now
                    
                except Exception as e:
                    print(f"Video receive ended: {e}")
                    break
        else:
            # data_only mode - wait for data channel messages until connection closes or timeout
            estimated_duration = (
                (total_frames_in_video / video_fps) if video_fps and video_fps > 0 else 60.0
            )
            wait_timeout = max(30.0, estimated_duration + 5.0)
            print(f"      Waiting for data messages (up to {wait_timeout:.1f}s)...")
            wait_start = time.time()
            last_report = wait_start
            while not peer_connection.closed_event.is_set() and time.time() - wait_start < wait_timeout:
                time.sleep(1)
                if time.time() - last_report >= 5:
                    print(
                        f"      Data messages received so far: {metrics['data_messages_received']}, "
                        f"unique frames: {len(metrics['unique_frame_ids'])}"
                    )
                    last_report = time.time()

            if peer_connection.closed_event.is_set():
                print("      ✓ Data collection complete")
            else:
                print("      ⚠️ Timed out waiting for data; closing connection")
                asyncio_loop.call_soon_threadsafe(peer_connection.closed_event.set)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    metrics["frames_sent"] = peer_connection.stream_track._frames_sent
    metrics["end_time"] = time.time()
    
    # Cleanup
    print(f"[7/7] Cleaning up...")
    if peer_connection.connectionState != "closed":
        asyncio.run_coroutine_threadsafe(peer_connection.close(), asyncio_loop).result()
    asyncio_loop.call_soon_threadsafe(asyncio_loop.stop)
    thread.join(timeout=2)
    
    elapsed = metrics["end_time"] - metrics["start_time"]
    print(f"      ✓ Test complete in {elapsed:.1f}s")
    print(f"      Frames sent: {metrics['frames_sent']}")
    print(f"      Video frames received: {metrics['video_frames_received']}")
    print(f"      Data messages received: {metrics['data_messages_received']}")
    print(f"      Unique frame IDs: {len(metrics['unique_frame_ids'])}")
    
    return metrics


def print_results(all_metrics: list[dict]):
    """Print comparison table."""
    print("\n" + "="*100)
    print("FRAME DELIVERY TEST RESULTS")
    print("="*100)
    
    # Header
    print(f"\n{'Mode':<15} {'Realtime':<10} {'Sent':<8} {'Video RX':<10} {'Data RX':<10} {'Unique IDs':<12} {'Complete%':<10} {'Dropped':<10}")
    print("-"*100)
    
    for m in all_metrics:
        mode = m["output_mode"]
        rt = "Yes" if m["realtime"] else "No"
        sent = m["frames_sent"]
        video_rx = m["video_frames_received"]
        data_rx = m["data_messages_received"]
        unique = len(m["unique_frame_ids"])
        total = m["total_frames_in_video"]
        complete_pct = (unique / total * 100) if total > 0 else 0
        dropped = total - unique
        
        print(f"{mode:<15} {rt:<10} {sent:<8} {video_rx:<10} {data_rx:<10} {unique:<12} {complete_pct:<10.1f} {dropped:<10}")
        print(f"  Connection states: {m.get('connection_states')}")
        print(f"  Data channel states: {m.get('data_channel_states')}")
    
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)
    
    # Find data_only modes
    data_only_rt = next((m for m in all_metrics if m["output_mode"] == "data_only" and m["realtime"]), None)
    data_only_buf = next((m for m in all_metrics if m["output_mode"] == "data_only" and not m["realtime"]), None)
    
    if data_only_rt and data_only_buf:
        total = data_only_rt["total_frames_in_video"]
        rt_unique = len(data_only_rt["unique_frame_ids"])
        buf_unique = len(data_only_buf["unique_frame_ids"])
        
        print(f"\nData Channel (realtime=True):  {rt_unique}/{total} frames ({rt_unique/total*100:.1f}%)")
        print(f"Data Channel (realtime=False): {buf_unique}/{total} frames ({buf_unique/total*100:.1f}%)")
        print(f"Difference: {abs(buf_unique - rt_unique)} frames")
        
        if buf_unique == total:
            print("\n✅ Data channel with buffering delivers 100% of frames!")
        elif buf_unique < total:
            print(f"\n⚠️  Even with buffering, {total - buf_unique} frames were lost via data channel")
        
        if rt_unique < buf_unique:
            print(f"📉 Realtime mode dropped {buf_unique - rt_unique} frames on data channel")


def parse_args():
    parser = argparse.ArgumentParser("Frame Delivery Test")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--inference-server-url", default="http://localhost:9001")
    parser.add_argument(
        "--stream-output",
        default=None,
        help="Specific workflow output to use for video stream (default: auto-detect first)",
    )
    parser.add_argument(
        "--data-output",
        default=None,
        help="Comma-separated workflow outputs for data channel (default: all). Use 'none' for none, 'all' for all.",
    )
    return parser.parse_args()


def main():
    print("🚀 Starting frame delivery test...", file=sys.stderr, flush=True)
    args = parse_args()
    print(f"✓ Arguments parsed", file=sys.stderr, flush=True)
    
    # Expand path
    video_path = os.path.expanduser(args.video_path)
    print(f"✓ Video path: {video_path}", file=sys.stderr, flush=True)
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    
    all_metrics = []
    
    # Test 1: data_only + realtime
    print("DEBUG: About to log TEST 1 message", file=sys.stderr, flush=True)
    print("\n\n🧪 STARTING TEST 1 OF 4: data_only + realtime=True\n")
    print("DEBUG: After log, calling run_test", file=sys.stderr, flush=True)
    metrics = run_test(
        video_path=video_path,
        workspace_id=args.workspace_id,
        workflow_id=args.workflow_id,
        api_key=args.api_key,
        inference_server_url=args.inference_server_url,
        output_mode="data_only",
        realtime=True,
        stream_output_param=args.stream_output,
        data_output_param=args.data_output,
    )
    all_metrics.append(metrics)
    print("\n⏳ Waiting 3 seconds before next test...\n")
    time.sleep(3)
    
    # Test 2: data_only + no realtime
    print("\n\n🧪 STARTING TEST 2 OF 4: data_only + realtime=False\n")
    metrics = run_test(
        video_path=video_path,
        workspace_id=args.workspace_id,
        workflow_id=args.workflow_id,
        api_key=args.api_key,
        inference_server_url=args.inference_server_url,
        output_mode="data_only",
        realtime=False,
        stream_output_param=args.stream_output,
        data_output_param=args.data_output,
    )
    all_metrics.append(metrics)
    print("\n⏳ Waiting 3 seconds before next test...\n")
    time.sleep(3)
    
    # Test 3: both + realtime
    print("\n\n🧪 STARTING TEST 3 OF 4: both + realtime=True\n")
    metrics = run_test(
        video_path=video_path,
        workspace_id=args.workspace_id,
        workflow_id=args.workflow_id,
        api_key=args.api_key,
        inference_server_url=args.inference_server_url,
        output_mode="both",
        realtime=True,
        stream_output_param=args.stream_output,
        data_output_param=args.data_output,
    )
    all_metrics.append(metrics)
    print("\n⏳ Waiting 3 seconds before next test...\n")
    time.sleep(3)
    
    # Test 4: both + no realtime
    print("\n\n🧪 STARTING TEST 4 OF 4: both + realtime=False\n")
    metrics = run_test(
        video_path=video_path,
        workspace_id=args.workspace_id,
        workflow_id=args.workflow_id,
        api_key=args.api_key,
        inference_server_url=args.inference_server_url,
        output_mode="both",
        realtime=False,
        stream_output_param=args.stream_output,
        data_output_param=args.data_output,
    )
    all_metrics.append(metrics)
    
    # Print results
    print_results(all_metrics)


if __name__ == "__main__":
    print("✓ Entering main block...", file=sys.stderr, flush=True)
    main()
    print("✓ Test suite complete!", file=sys.stderr, flush=True)

