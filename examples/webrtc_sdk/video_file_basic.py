"""
Minimal sample using the SDK's WebRTC namespace to stream video file frames
to a running inference server with WebRTC worker enabled.

Usage:
  python examples/webrtc_sdk/video_file_basic.py \\
      --video-path /path/to/video.mp4 \\
      --workspace-name <your_workspace> \\
      --workflow-id <your_workflow_id> \\
      [--api-url http://localhost:9001] \\
      [--api-key <ROBOFLOW_API_KEY>] \\
      [--stream-output <output_name>] \\
      [--data-output <output_name>]

Press 'q' in the preview window to exit.
"""
import argparse

import av
import cv2

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import StreamConfig, VideoFileSource

# Suppress FFmpeg warnings from PyAV
av.logging.set_level(av.logging.ERROR)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("WebRTC SDK video_file_basic")
    p.add_argument(
        "--video-path", required=True, help="Path to video file to process"
    )
    p.add_argument("--api-url", default="https://serverless.roboflow.com")
    p.add_argument("--workspace-name", required=True)
    p.add_argument("--workflow-id", required=True)
    p.add_argument("--image-input-name", default="image")
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--stream-output",
        default=None,
        help="Name of the workflow output to stream (e.g., 'image_output')",
    )
    p.add_argument(
        "--data-output",
        default=None,
        help="Name of the workflow output to receive via data channel",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = InferenceHTTPClient.init(api_url=args.api_url, api_key=args.api_key)

    # Prepare source
    source = VideoFileSource(args.video_path)

    # Prepare config
    stream_output = [args.stream_output] if args.stream_output else []
    data_output = [args.data_output] if args.data_output else []
    config = StreamConfig(stream_output=stream_output, data_output=data_output)

    # Start streaming session
    with client.webrtc.stream(
        source=source,
        workflow=args.workflow_id,
        workspace=args.workspace_name,
        image_input=args.image_input_name,
        config=config,
    ) as session:
        # Register data handler to print messages
        @session.data.on_data()
        def on_message(msg):  # noqa: ANN001
            print(msg)

        # Stream video from the server (if stream_output is configured)
        for frame in session.video():
            cv2.imshow("WebRTC SDK - Video File", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
