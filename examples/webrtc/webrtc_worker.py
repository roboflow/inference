import argparse
import asyncio
import json
import logging
import sys
import urllib.parse
from pathlib import Path
from threading import Event, Thread
from typing import Optional, Union

import cv2 as cv
import numpy as np
import requests
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame
from av.logging import ERROR, set_libav_level

from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCData,
    WebRTCOffer,
    WebRTCTURNConfig,
    WorkflowConfiguration,
)
from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest
from inference.core.roboflow_api import get_workflow_specification
from inference.core.utils.async_utils import Queue

logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s - [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(Path(__file__).stem)


class FramesGrabber:
    def __init__(
        self,
        source_path: Union[int, str],
    ):
        self._cap = cv.VideoCapture(source_path)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam")
        self._fps = self._cap.get(cv.CAP_PROP_FPS)

    def get_frame(self) -> Optional[np.ndarray]:
        ret, np_frame = self._cap.read()
        if not ret:
            return None
        return np_frame

    def get_fps(self) -> Optional[float]:
        return self._fps


class StreamTrack(VideoStreamTrack):
    def __init__(
        self,
        asyncio_loop: Optional[asyncio.AbstractEventLoop] = None,
        source_path: Optional[Union[int, str]] = None,
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

    def set_track(self, track: RemoteStreamTrack):
        self.track = track
        self._recv_task = self._loop.create_task(self._recv_loop(), name="recv_loop")

    async def stop_recv_loop(self):
        if self._recv_task:
            logger.info("Cancelling WebRTC recv loop")
            self._recv_task.cancel()
            self._recv_task = None
        await self.recv_queue.async_put(None)

    async def _recv_loop(self):
        logger.info("Starting WebRTC recv loop")
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            set_libav_level(ERROR)
            self._av_logging_set = True

        try:
            while self.track.readyState != "ended":
                frame: VideoFrame = await self.track.recv()
                await self.recv_queue.async_put(frame)

        except asyncio.CancelledError:
            logger.info("WebRTC recv loop cancelled")
        except MediaStreamError:
            if not self.complete:
                logger.error("WebRTC recv loop finished due to MediaStreamError")
        except Exception as exc:
            logger.error("Error in WebRTC recv loop: %s", exc)

        await self.recv_queue.async_put(None)

    async def recv(self):
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            set_libav_level(ERROR)
            self._av_logging_set = True

        if self._source is None:
            return

        np_frame = await self._loop.run_in_executor(
            None,
            self._source.get_frame,
        )
        if np_frame is None:
            logger.info("%s: No more frames", self.__class__.__name__)
            await self.stop_recv_loop()
            raise MediaStreamError("No more frames")

        new_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")
        new_frame.pts, new_frame.time_base = await self.next_timestamp()

        return new_frame


class RTCPeerConnectionWithDataChannel(RTCPeerConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_channel: Optional[RTCDataChannel] = None
        self.stream_track: Optional[StreamTrack] = None
        self.closed_event: Event = Event()


async def init_rtc_peer_connection_with_local_description(
    asyncio_loop: asyncio.AbstractEventLoop,
    webrtc_turn_config: Optional[WebRTCTURNConfig] = None,
    source: Optional[str] = None,
) -> RTCPeerConnectionWithDataChannel:
    if webrtc_turn_config:
        turn_server = RTCIceServer(
            urls=[webrtc_turn_config.urls],
            username=webrtc_turn_config.username,
            credential=webrtc_turn_config.credential,
        )
        peer_connection = RTCPeerConnectionWithDataChannel(
            configuration=RTCConfiguration(iceServers=[turn_server]),
        )
    else:
        peer_connection = RTCPeerConnectionWithDataChannel()

    is_rtmp = is_rtmp_url(url=source)
    if is_rtmp:
        logger.info("Requesting processing of RTMP/RTSP stream: %s", source)
        stream_track = StreamTrack(
            asyncio_loop=asyncio_loop,
        )
        peer_connection.addTransceiver("video", direction="recvonly")
        peer_connection.stream_track = stream_track
    else:
        logger.info(
            "Requesting processing of local video stream: %s",
            source if source else "webcam",
        )
        if source is None:
            source = 0
        stream_track = StreamTrack(
            asyncio_loop=asyncio_loop,
            source_path=source,
        )
        peer_connection.addTrack(stream_track)

    relay = MediaRelay()

    @peer_connection.on("track")
    def on_track(track: RemoteStreamTrack):
        logger.info("track received")
        stream_track.set_track(track=relay.subscribe(track))
        peer_connection.stream_track = stream_track

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("connection state: %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            logger.info("Stopping recv loop")
            await stream_track.stop_recv_loop()
            if stream_track.track:
                logger.info("Stopping track")
                stream_track.track.stop()
            peer_connection.closed_event.set()
            logger.info("Stopping peer connection")
            await peer_connection.close()

    data_channel = peer_connection.createDataChannel("inference")

    @data_channel.on("message")
    def on_message(message):
        print(message)

    peer_connection.data_channel = data_channel

    offer: RTCSessionDescription = await peer_connection.createOffer()
    await peer_connection.setLocalDescription(offer)
    while peer_connection.iceGatheringState != "complete":
        logger.debug("Waiting for ice gathering to complete")
        await asyncio.sleep(0.1)

    return peer_connection


def is_rtmp_url(url: str) -> bool:
    return str(url).lower().startswith("rtmp:") or str(url).lower().startswith("rtsp:")


class MustBeFileOrRTSP(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values.strip() or (
            not Path(values.strip()).exists() and not is_rtmp_url(values.strip())
        ):
            raise argparse.ArgumentError(
                argument=self, message="Expected file path or RTSP/RTMP url"
            )
        setattr(namespace, self.dest, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream video file or webcam to Roboflow for processing, or request processed RTSP/RTMP stream"
    )
    parser.add_argument(
        "--source",
        required=False,
        type=str,
        default=None,
        action=MustBeFileOrRTSP,
        help="RTSP/RTMP url or path to video file, if not provided webcam will be used",
    )
    parser.add_argument("--workflow-id", required=True, type=str)
    parser.add_argument("--workspace-id", required=True, type=str)
    parser.add_argument("--inference-server-url", required=True, type=str)
    parser.add_argument("--api-key", required=True, type=str)
    parser.add_argument("--realtime", required=False, action="store_true")
    parser.add_argument("--turn-url", required=False, type=str)
    parser.add_argument("--turn-username", required=False, type=str)
    parser.add_argument("--turn-credential", required=False, type=str)
    parser.add_argument(
        "--output-mode",
        required=False,
        type=str,
        default="both",
        choices=["data_only", "video_only", "both"],
        help="Output mode: data_only (JSON only), video_only (video only), both (default)",
    )
    parser.add_argument(
        "--stream-output",
        required=False,
        type=str,
        default=None,
        help="Which workflow output to use for video stream (auto-detected if not specified)",
    )
    parser.add_argument(
        "--data-outputs",
        required=False,
        type=str,
        default=None,
        help="Comma-separated list of workflow outputs for data channel (e.g., 'predictions,count'). Use 'all' for all outputs, or omit for all outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Starting WebRTC worker with output_mode: {args.output_mode}")
    if args.output_mode == "data_only":
        logger.info("DATA_ONLY mode: Server will send JSON data via data channel only (no video track)")
    elif args.output_mode == "video_only":
        logger.info("VIDEO_ONLY mode: Server will send processed video only (no data channel messages)")
    elif args.output_mode == "both":
        logger.info("BOTH mode: Server will send both video and JSON data")

    workflow_specification = get_workflow_specification(
        api_key=args.api_key,
        workspace_id=args.workspace_id,
        workflow_id=args.workflow_id,
    )

    # Find available outputs
    workflow_outputs = workflow_specification.get("outputs", [])
    available_output_names = [o.get("name") for o in workflow_outputs]

    if not workflow_outputs:
        logger.warning("⚠️  Workflow has no outputs defined")
    else:
        logger.info(f"Available workflow outputs: {available_output_names}")

    # Determine stream_output
    stream_output_to_use = None
    if args.output_mode in ["both", "video_only"]:
        if args.stream_output:
            # User specified stream output - validate it exists
            if args.stream_output not in available_output_names:
                raise ValueError(
                    f"❌ stream_output '{args.stream_output}' not found in workflow outputs. "
                    f"Available: {available_output_names}"
                )
            stream_output_to_use = args.stream_output
            logger.info(f"Using specified stream_output: {stream_output_to_use}")
        else:
            # Let backend auto-detect first valid image output
            logger.info("stream_output not specified, backend will auto-detect first valid image output")

    # Determine data_output
    data_output_to_use = None
    if args.data_outputs:
        if args.data_outputs.lower() == "all":
            data_output_to_use = None  # None = all outputs
            logger.info("data_output: ALL outputs (None)")
        elif args.data_outputs.lower() == "none":
            data_output_to_use = []  # Empty = no data
            logger.info("data_output: NO outputs ([])")
        else:
            # Parse comma-separated list
            requested_fields = [f.strip() for f in args.data_outputs.split(",")]

            # Validate all requested fields exist
            invalid_fields = [f for f in requested_fields if f not in available_output_names]
            if invalid_fields:
                raise ValueError(
                    f"❌ data_output fields {invalid_fields} not found in workflow outputs. "
                    f"Available: {available_output_names}"
                )

            data_output_to_use = requested_fields
            logger.info(f"data_output: {data_output_to_use}")
    else:
        # Default: send all outputs
        data_output_to_use = None
        logger.info("data_output: ALL outputs (default)")

    webrtc_turn_config = None
    if args.turn_url:
        webrtc_turn_config = WebRTCTURNConfig(
            urls=args.turn_url,
            username=args.turn_username,
            credential=args.turn_credential,
        )

    asyncio_loop = asyncio.new_event_loop()
    loop_thread = Thread(target=asyncio_loop.run_forever, daemon=True)
    loop_thread.start()
    asyncio.set_event_loop(asyncio_loop)
    future = asyncio.run_coroutine_threadsafe(
        init_rtc_peer_connection_with_local_description(
            asyncio_loop=asyncio_loop,
            webrtc_turn_config=webrtc_turn_config,
            source=args.source,
        ),
        asyncio_loop,
    )
    peer_connection = future.result()

    request = WebRTCWorkerRequest(
        api_key=args.api_key,
        workflow_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration",
            workflow_id=args.workflow_id,
            workspace_name=args.workspace_id,
            image_input_name="image",
            workflows_parameters={},
            workflows_thread_pool_workers=4,
            cancel_thread_pool_tasks_on_exit=True,
            video_metadata_input_name="video_metadata",
        ),
        webrtc_offer=WebRTCOffer(
            type=peer_connection.localDescription.type,
            sdp=peer_connection.localDescription.sdp,
        ),
        webrtc_turn_config=webrtc_turn_config,
        output_mode=args.output_mode,
        stream_output=[stream_output_to_use] if stream_output_to_use else [],
        data_output=data_output_to_use,
        webrtc_realtime_processing=args.realtime,
        rtsp_url=args.source if is_rtmp_url(args.source) else None,
    )

    https_verify = True
    if args.inference_server_url.startswith("https://") and (
        "localhost" in args.inference_server_url
        or "127.0.0.1" in args.inference_server_url
    ):
        https_verify = False

    response = requests.post(
        urllib.parse.urljoin(args.inference_server_url, "initialise_webrtc_worker"),
        json=request.model_dump(),
        verify=https_verify,
    )
    webrtc_answer = response.json()
    if response.status_code != 200:
        raise Exception(f"Failed to initialise WebRTC pipeline: {response.text}")

    # Set up data channel listener for JSON data
    data_channel_message_count = [0]  # Use list for closure
    if peer_connection.data_channel:
        @peer_connection.data_channel.on("message")
        def on_data_message(message):
            data_channel_message_count[0] += 1
            try:
                data = json.loads(message)
                logger.info(f"=== Data Channel Message #{data_channel_message_count[0]} ===")
                logger.info(f"Frame ID: {data.get('video_metadata', {}).get('frame_id')}")

                if data.get('serialized_output_data'):
                    logger.info(f"Output fields: {list(data['serialized_output_data'].keys())}")
                    for field, value in data['serialized_output_data'].items():
                        if isinstance(value, str) and value.startswith('data:image'):
                            logger.info(f"  {field}: <base64 image, {len(value)} bytes>")
                        elif isinstance(value, dict) and 'predictions' in value:
                            logger.info(f"  {field}: {len(value.get('predictions', []))} detections")
                        else:
                            logger.info(f"  {field}: {value}")
                else:
                    logger.info("  No output data")

                if data.get('errors'):
                    logger.warning(f"Errors: {data['errors']}")

            except json.JSONDecodeError:
                logger.error(f"Failed to parse message: {message[:200]}...")

    future = asyncio.run_coroutine_threadsafe(
        peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=webrtc_answer["sdp"], type=webrtc_answer["type"])
        ),
        asyncio_loop,
    )
    future.result()

    # Track active data output mode: "all" (None), "none" ([]), or list of field names
    active_data_fields = []  # Initialize for custom mode
    if data_output_to_use is None:
        active_data_mode = "all"  # "all" means None
    elif data_output_to_use == []:
        active_data_mode = "none"  # "none" means []
    else:
        active_data_mode = "custom"  # Custom list
        active_data_fields = list(data_output_to_use)  # Copy of active fields

    def draw_mode_indicator(frame, mode_text):
        """Draw mode indicator overlay (top-left)"""
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        # Get text size to draw proper background
        (text_width, text_height), baseline = cv.getTextSize(mode_text, font, font_scale, font_thickness)

        # Draw background rectangle
        padding = 10
        bg_x1, bg_y1 = 10, 10
        bg_x2, bg_y2 = 10 + text_width + padding * 2, 10 + text_height + padding * 2

        cv.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

        # Draw text
        text_x = bg_x1 + padding
        text_y = bg_y1 + padding + text_height
        cv.putText(
            frame,
            mode_text,
            (text_x, text_y),
            font,
            font_scale,
            (100, 255, 100),  # Brighter green
            font_thickness,
            cv.LINE_AA
        )

        return frame

    def draw_controls_hint(frame, controls_text):
        """Draw controls hint overlay (bottom)"""
        font = cv.FONT_HERSHEY_SIMPLEX
        controls_font_scale = 0.45
        controls_thickness = 1

        h = frame.shape[0]

        (ctrl_width, ctrl_height), ctrl_baseline = cv.getTextSize(
            controls_text, font, controls_font_scale, controls_thickness
        )

        # Draw background for controls
        ctrl_padding = 8
        ctrl_bg_x1 = 10
        ctrl_bg_y1 = h - ctrl_height - ctrl_padding * 2 - 10
        ctrl_bg_x2 = ctrl_bg_x1 + ctrl_width + ctrl_padding * 2
        ctrl_bg_y2 = h - 10

        cv.rectangle(frame, (ctrl_bg_x1, ctrl_bg_y1), (ctrl_bg_x2, ctrl_bg_y2), (0, 0, 0), -1)

        # Draw controls text
        ctrl_text_x = ctrl_bg_x1 + ctrl_padding
        ctrl_text_y = ctrl_bg_y2 - ctrl_padding - ctrl_baseline
        cv.putText(
            frame,
            controls_text,
            (ctrl_text_x, ctrl_text_y),
            font,
            controls_font_scale,
            (200, 200, 200),  # Light gray
            controls_thickness,
            cv.LINE_AA
        )

        return frame

    def draw_output_list(frame, available_outputs, current_mode, active_fields=None):
        """Draw list of available outputs with active indicators"""
        font = cv.FONT_HERSHEY_SIMPLEX
        x_start = 10
        y_start = 80
        line_height = 22

        # Title
        if current_mode == "all":
            title = "Data Outputs (ALL)"
            title_color = (100, 255, 100)
        elif current_mode == "none":
            title = "Data Outputs (NONE)"
            title_color = (100, 100, 100)
        else:
            title = f"Data Outputs ({len(active_fields)} active)"
            title_color = (100, 200, 255)

        cv.putText(frame, title, (x_start, y_start), font, 0.5, title_color, 1, cv.LINE_AA)
        y_start += line_height + 5

        # Draw each output
        for i, output in enumerate(available_outputs):
            key_letter = chr(ord('a') + i) if i < 26 else '?'
            output_name = output.get('name', 'unnamed')

            # Determine if active
            if current_mode == "all":
                is_active = True
            elif current_mode == "none":
                is_active = False
            else:
                is_active = output_name in active_fields

            # Format line with ASCII checkbox
            indicator = "[X]" if is_active else "[ ]"
            color = (100, 255, 100) if is_active else (100, 100, 100)
            text = f"  [{key_letter}] {indicator} {output_name}"

            cv.putText(frame, text, (x_start, y_start + i * line_height), font, 0.45, color, 1, cv.LINE_AA)

        # Controls
        y_controls = y_start + len(available_outputs) * line_height + 10
        cv.putText(frame, "  [+] All  [-] None", (x_start, y_controls), font, 0.45, (200, 200, 200), 1, cv.LINE_AA)

        return frame

    def handle_keyboard_input(key: int) -> bool:
        nonlocal active_data_mode

        if key == -1:
            return True

        if key == ord("q"):
            logger.info("Quitting")
            return False

        # Check data channel status for all commands except quit
        if not peer_connection.data_channel or peer_connection.data_channel.readyState != "open":
            logger.error("Data channel not open")
            return True

        # Handle + key (all outputs)
        if key == ord("+") or key == ord("="):
            logger.info("Setting data_output to ALL (None)")
            active_data_mode = "all"
            message = json.dumps(
                WebRTCData(
                    stream_output=None,
                    data_output=None,
                ).model_dump()
            )
            peer_connection.data_channel.send(message)
            return True

        # Handle - key (no outputs)
        if key == ord("-"):
            logger.info("Setting data_output to NONE ([])")
            active_data_mode = "none"
            message = json.dumps(
                WebRTCData(
                    stream_output=None,
                    data_output=[],
                ).model_dump()
            )
            peer_connection.data_channel.send(message)
            return True

        # Handle 0-9 keys (stream output selection)
        if chr(key) in "1234567890":
            if chr(key) == "0":
                message = json.dumps(
                    WebRTCData(
                        stream_output="",
                        data_output=None,
                    ).model_dump()
                )
                logger.info("Turning off stream output via data channel")
            else:
                max_ind = max(0, len(workflow_specification.get("outputs", [])) - 1)
                output_ind = min(key - ord("1"), max_ind)
                output_name = workflow_specification.get("outputs")[output_ind].get(
                    "name", ""
                )
                message = json.dumps(
                    WebRTCData(
                        stream_output=output_name,
                        data_output=None,
                    ).model_dump()
                )
                logger.info("Setting stream output via data channel: %s", output_name)
            peer_connection.data_channel.send(message)
            return True

        # Handle a-z toggle (individual field toggle)
        if chr(key).isalpha() and chr(key).lower() in "abcdefghijklmnopqrstuvwxyz":
            key_index = ord(chr(key).lower()) - ord("a")
            if key_index < len(workflow_outputs):
                output_name = workflow_outputs[key_index].get("name", "")

                # Toggle logic
                if active_data_mode == "all":
                    # Was "all", switch to custom with all except this one
                    active_data_mode = "custom"
                    active_data_fields.clear()
                    active_data_fields.extend([o.get("name") for o in workflow_outputs])
                    active_data_fields.remove(output_name)
                    logger.info(f"Toggled OFF '{output_name}' (was ALL)")
                elif active_data_mode == "none":
                    # Was "none", enable only this field
                    active_data_mode = "custom"
                    active_data_fields.clear()
                    active_data_fields.append(output_name)
                    logger.info(f"Toggled ON '{output_name}' (was NONE)")
                else:
                    # Custom mode - toggle
                    if output_name in active_data_fields:
                        active_data_fields.remove(output_name)
                        logger.info(f"Toggled OFF '{output_name}'")
                    else:
                        active_data_fields.append(output_name)
                        logger.info(f"Toggled ON '{output_name}'")

                # Send updated list directly as array
                logger.info(f"Active fields: {active_data_fields}")
                message = json.dumps(
                    WebRTCData(
                        stream_output=None,
                        data_output=active_data_fields if active_data_fields else [],
                    ).model_dump()
                )
                peer_connection.data_channel.send(message)
            return True

        return True

    # For data_only mode, show blank window with data controls
    if args.output_mode == "data_only":
        logger.info("DATA_ONLY mode: Showing placeholder window with output controls")

        try:
            while not peer_connection.closed_event.is_set():
                # Create black frame with overlays
                frame = np.zeros((520, 700, 3), dtype=np.uint8)

                mode_text = f"MODE: {args.output_mode.upper()}"
                frame = draw_mode_indicator(frame, mode_text)

                if active_data_mode == "custom":
                    frame = draw_output_list(frame, workflow_outputs, active_data_mode, active_data_fields)
                else:
                    frame = draw_output_list(frame, workflow_outputs, active_data_mode, None)

                controls_text = "+ = all | - = none | a-z = data | q = quit"
                frame = draw_controls_hint(frame, controls_text)

                cv.imshow("WebRTC Worker - Interactive Mode", frame)
                key = cv.waitKey(100)  # 100ms delay to keep UI responsive

                # Handle keyboard input
                should_continue = handle_keyboard_input(key)
                if not should_continue:
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
    else:
        # For modes with video, use the video frame loop
        while not peer_connection.closed_event.is_set():
            frame: Optional[VideoFrame] = peer_connection.stream_track.recv_queue.sync_get()
            if frame is None:
                logger.info("No more frames")
                break

            # Convert frame to numpy
            np_frame = frame.to_ndarray(format="bgr24")

            # Draw overlays
            mode_text = f"MODE: {args.output_mode.upper()}"
            np_frame = draw_mode_indicator(np_frame, mode_text)

            if active_data_mode == "custom":
                np_frame = draw_output_list(np_frame, workflow_outputs, active_data_mode, active_data_fields)
            else:
                np_frame = draw_output_list(np_frame, workflow_outputs, active_data_mode, None)

            controls_text = "+ = all data | - = no data | 0-9 = stream | a-z = data | q = quit"
            np_frame = draw_controls_hint(np_frame, controls_text)

            cv.imshow("WebRTC Worker - Interactive Mode", np_frame)
            key = cv.waitKey(1)

            # Handle keyboard input
            should_continue = handle_keyboard_input(key)
            if not should_continue:
                break

    # Cleanup
    cv.destroyAllWindows()  # Close OpenCV windows (works for all modes now)

    if args.output_mode != "data_only":
        asyncio.run_coroutine_threadsafe(
            peer_connection.stream_track.stop_recv_loop(),
            asyncio_loop,
        ).result()

    if peer_connection.connectionState != "closed":
        logger.info("Closing WebRTC connection")
        asyncio.run_coroutine_threadsafe(
            peer_connection.close(),
            asyncio_loop,
        ).result()
    logger.info("Stopping asyncio loop")
    asyncio_loop.call_soon_threadsafe(asyncio_loop.stop)
    loop_thread.join(timeout=5)
    asyncio_loop.close()


if __name__ == "__main__":
    main()
