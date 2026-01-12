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
      [--data-output <output_name>] \\
      [--file-output /path/to/output.mp4]

Press 'q' in the preview window to exit.
"""
import argparse

import av
import cv2

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import StreamConfig, VideoFileSource, VideoMetadata

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
    p.add_argument(
        "--file-output",
        default=None,
        help="Path to save output video file (optional)",
    )
    p.add_argument(
        "--realtime-processing",
        action="store_true",
        help="Process at original video FPS (default: process as fast as possible)",
    )
    return p.parse_args()


def get_video_fps(video_path: str) -> float:
    """Get FPS from source video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0

def upload_progress(uploaded_chunks: int, total_chunks: int) -> None:
    print(f"Upload progress: {uploaded_chunks} / {total_chunks}")

def main() -> None:
    args = parse_args()
    client = InferenceHTTPClient.init(api_url=args.api_url, api_key=args.api_key)

    # Prepare source
    source = VideoFileSource(
        args.video_path,
        on_upload_progress=upload_progress,
        # use_datachannel_frames=False,  # Use video track for lower bandwidth
        realtime_processing=args.realtime_processing,
    )

    # Prepare config
    stream_output = [args.stream_output] if args.stream_output else []
    data_output = [args.data_output] if args.data_output else []
    config = StreamConfig(
        stream_output=stream_output,
        data_output=data_output,
        realtime_processing=args.realtime_processing,
    )

    # Create streaming session
    session = client.webrtc.stream(
        source=source,
        workflow=args.workflow_id,
        workspace=args.workspace_name,
        image_input=args.image_input_name,
        config=config,
    )

    # Video writer state (for optional file output)
    writer = None
    fps = get_video_fps(args.video_path) if args.file_output else None

    # Register frame handler
    @session.on_frame
    def show_frame(frame, metadata):
        nonlocal writer
        # Save to file if output path specified
        if args.file_output:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.file_output, fourcc, fps, (w, h))
            writer.write(frame)

        cv2.imshow("WebRTC SDK - Video File", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            session.close()  # Close session and cleanup resources

    # Register data handlers
    # Global handler (receives entire serialized_output_data dict + metadata)
    @session.on_data()
    def on_message(data: dict, metadata: VideoMetadata):
        print(f"Frame {metadata.frame_id}: {data}")

    # Field-specific handler example (uncomment and customize based on your workflow):
    # @session.on_data("predictions")
    # def on_predictions(predictions: dict, metadata: VideoMetadata):
    #     print(f"Frame {metadata.frame_id} predictions: {predictions}")

    # Run the session (auto-starts, blocks until close() is called or stream ends)
    # Automatically closes on exception or when stream ends
    session.run()

    # Cleanup video writer
    if writer:
        writer.release()
        print(f"Saved output to {args.file_output}")


if __name__ == "__main__":
    main()
