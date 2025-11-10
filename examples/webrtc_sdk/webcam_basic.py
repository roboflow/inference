"""
Minimal sample using the SDK's WebRTC namespace to stream webcam frames
to a running inference server with WebRTC worker enabled.

Usage:
  python examples/webrtc_sdk/webcam_basic.py \
      --api-url http://localhost:9001 \
      --workspace-name <your_workspace> \
      --workflow-id <your_workflow_id> \
      --image-input-name image \
      [--api-key <ROBOFLOW_API_KEY>] \
      [--stream-output <output_name>]

Press 'q' in the preview window to exit.
"""
import argparse

import cv2

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc.config import WebcamConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("WebRTC SDK webcam_basic")
    p.add_argument("--api-url", required=False, default="https://serverless.roboflow.com")
    p.add_argument("--workspace-name", required=True)
    p.add_argument("--workflow-id", required=True)
    p.add_argument("--image-input-name", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--stream-output", default=None, help="Name of the workflow output to stream (e.g., 'image_output')")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = InferenceHTTPClient.init(api_url=args.api_url, api_key=args.api_key)

    resolution = None
    if args.width and args.height:
        resolution = (args.width, args.height)

    stream_output = [args.stream_output] if args.stream_output else []
    cfg = WebcamConfig(resolution=resolution, stream_output=stream_output)

    with client.webrtc.use_webcam(
        image_input_name=args.image_input_name,
        workspace_name=args.workspace_name,
        workflow_id=args.workflow_id,
        config=cfg,
    ) as s:
        # If you want to also consume inference data messages, register a handler:
        @s.data.on("message")
        def on_message(msg):  # noqa: ANN001
            # Print raw message (JSON string from server)
            print(msg)

        for frame in s.video.stream():
            cv2.imshow("WebRTC SDK - Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
