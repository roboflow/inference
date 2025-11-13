"""
Example showing callback-based workflow processing with WebRTC.

This demonstrates the InferencePipeline-like API for WebRTC streaming,
where you provide a single callback function that receives all workflow
outputs including the decoded image frame.

Usage:
  python examples/webrtc_sdk/workflow_webrtc_callback.py \
      --workspace-name <your_workspace> \
      --workflow-id <your_workflow_id> \
      [--api-url http://localhost:9001] \
      [--api-key <ROBOFLOW_API_KEY>] \
      [--video-source 0]  # 0 for webcam, or path to video file

Press 'q' in the preview window to exit.
"""

import argparse

import cv2

from inference_sdk import InferenceHTTPClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("WebRTC SDK workflow callback example")
    p.add_argument("--api-url", default="https://serverless.roboflow.com")
    p.add_argument("--workspace-name", required=True)
    p.add_argument("--workflow-id", required=True)
    p.add_argument("--image-input-name", default="image")
    p.add_argument("--image-output-name", default="image")
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--video-source",
        default="0",
        help="Video source: integer for webcam (e.g., 0) or path to video file",
    )
    p.add_argument(
        "--workflows-parameters",
        default=None,
        help="Optional workflow parameters as key=value pairs (e.g., 'model=yolov8,confidence=0.5')",
    )
    return p.parse_args()


def parse_video_source(source_str: str):
    """Parse video source string to int (webcam) or str (file path)."""
    try:
        # Try to parse as integer (webcam ID)
        return int(source_str)
    except ValueError:
        # It's a file path
        return source_str


def parse_workflow_parameters(params_str: str) -> dict:
    """Parse workflow parameters from command line string.

    Example: "model=yolov8,confidence=0.5" -> {"model": "yolov8", "confidence": "0.5"}
    """
    if not params_str:
        return {}

    params = {}
    for pair in params_str.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            params[key.strip()] = value.strip()
    return params


def main() -> None:
    args = parse_args()

    # Initialize HTTP client
    client = InferenceHTTPClient.init(api_url=args.api_url, api_key=args.api_key)

    # Parse video source
    video_source = parse_video_source(args.video_source)

    # Parse workflow parameters if provided
    workflow_params = None
    if args.workflows_parameters:
        workflow_params = parse_workflow_parameters(args.workflows_parameters)

    # Define callback function that will be called for each processed frame
    def on_prediction(data: dict, metadata) -> None:
        """
        Callback invoked for each processed frame.

        Args:
            data: Dictionary containing:
                - "image": Decoded numpy array (automatically decoded from base64)
                - Other workflow outputs (predictions, etc.)
            metadata: VideoMetadata with frame_id, pts, timestamp, etc.
        """
        # Get the decoded image (already numpy array, not base64)
        frame = data.get(args.image_output_name)

        if frame is not None:
            # Display the frame
            cv2.imshow("WebRTC Workflow - Callback API", frame)

            # Check for 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                pipeline.terminate()

        # Print frame info and workflow outputs
        print(f"Frame {metadata.frame_id} - PTS: {metadata.pts}")

        # Print other workflow outputs (excluding the image to keep it clean)
        for key, value in data.items():
            if key != args.image_output_name:
                print(f"  {key}: {value}")

    # Create pipeline with callback
    pipeline = client.start_inference_pipeline_with_workflow_webrtc(
        video_reference=video_source,
        workspace_name=args.workspace_name,
        workflow_id=args.workflow_id,
        on_prediction=on_prediction,
        image_input_name=args.image_input_name,
        image_output_name=args.image_output_name,
        workflows_parameters=workflow_params,
    )

    print(f"Starting WebRTC workflow pipeline...")
    print(f"  Workspace: {args.workspace_name}")
    print(f"  Workflow: {args.workflow_id}")
    print(f"  Video source: {video_source}")
    print(f"  Press 'q' in the video window to exit")

    try:
        # Start pipeline in background thread (non-blocking)
        pipeline.start()

        # Wait for the pipeline to finish
        # (it will run until terminated by callback or error)
        while pipeline.is_running():
            import time
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up
        pipeline.terminate()
        cv2.destroyAllWindows()
        print("Pipeline terminated")


if __name__ == "__main__":
    main()
