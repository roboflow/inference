"""
WebRTC SDK example demonstrating DATA_ONLY output mode.

This example shows how to use the DATA_ONLY mode to receive only inference
results without video feedback, which significantly reduces bandwidth usage.

DATA_ONLY mode is ideal for:
- Analytics and metrics collection
- Headless inference servers
- High-throughput object counting
- Logging detections for later analysis
- IoT devices with limited bandwidth

Usage:
  python examples/webrtc_sdk/data_only_example.py \
      --workspace-name <your_workspace> \
      --workflow-id <your_workflow_id> \
      [--api-url http://localhost:9001] \
      [--api-key <ROBOFLOW_API_KEY>] \
      [--duration 30]

Press Ctrl+C to stop early.
"""
import argparse
import time
from collections import defaultdict
from datetime import datetime

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import OutputMode, StreamConfig, VideoMetadata, WebcamSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("WebRTC SDK Data-Only Mode Example")
    p.add_argument("--api-url", default="http://localhost:9001")
    p.add_argument("--workspace-name", required=True)
    p.add_argument("--workflow-id", required=True)
    p.add_argument("--image-input-name", default="image")
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--duration",
        type=int,
        default=30,
        help="How long to run in seconds (default: 30)",
    )
    p.add_argument(
        "--data-fields",
        type=str,
        default=None,
        help="Comma-separated list of fields to receive (default: all outputs)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = InferenceHTTPClient.init(api_url=args.api_url, api_key=args.api_key)

    # Prepare source
    source = WebcamSource()

    # Configure data output fields
    if args.data_fields:
        data_output = [f.strip() for f in args.data_fields.split(",")]
    else:
        data_output = []  # Empty list means all outputs

    # Configure for DATA_ONLY mode - no video will be sent back
    config = StreamConfig(
        output_mode=OutputMode.DATA_ONLY,  # Only data, no video
        data_output=data_output,  # What fields to receive
        realtime_processing=True,  # Process frames in realtime
    )

    # Statistics tracking
    stats = {
        "frames_processed": 0,
        "start_time": time.time(),
        "detections_per_frame": [],
        "field_counts": defaultdict(int),
    }

    print("\n" + "=" * 70)
    print("WebRTC SDK - DATA_ONLY Mode Example")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Output Mode: DATA_ONLY (no video feedback)")
    print(f"  Data Fields: {data_output if data_output else 'ALL workflow outputs'}")
    print(f"  Duration: {args.duration} seconds")
    print(f"  API URL: {args.api_url}")
    print(f"\nStarting session... (Press Ctrl+C to stop early)")
    print("-" * 70 + "\n")

    # Start streaming session
    with client.webrtc.stream(
        source=source,
        workflow=args.workflow_id,
        workspace=args.workspace_name,
        image_input=args.image_input_name,
        config=config,
    ) as session:

        # Global data handler - receives all workflow outputs
        @session.on_data()
        def handle_all_data(data: dict, metadata: VideoMetadata):
            stats["frames_processed"] += 1
            frame_num = stats["frames_processed"]

            # Track which fields we received
            if data:
                for field_name in data.keys():
                    stats["field_counts"][field_name] += 1

            # Print periodic updates with property_definition value
            if frame_num % 10 == 0:
                elapsed = time.time() - stats["start_time"]
                fps = frame_num / elapsed if elapsed > 0 else 0

                # Extract property_definition value if present
                property_value = data.get("property_definition", "N/A") if data else "N/A"

                print(
                    f"Frame {frame_num:4d} | "
                    f"FPS: {fps:5.1f} | "
                    f"property_definition: {property_value} | "
                    f"Fields: {list(data.keys()) if data else 'none'}"
                )

        # Field-specific handler for predictions (if available)
        @session.on_data("predictions")
        def handle_predictions(predictions: dict, metadata: VideoMetadata):
            # Count detections
            if isinstance(predictions, dict) and "predictions" in predictions:
                num_detections = len(predictions["predictions"])
                stats["detections_per_frame"].append(num_detections)

                # Log significant events
                if num_detections > 5:
                    print(f"  → High activity: {num_detections} detections!")

        # Run for specified duration
        start_time = time.time()
        try:
            while time.time() - start_time < args.duration:
                time.sleep(0.1)  # Small sleep to prevent busy loop
        except KeyboardInterrupt:
            print("\n\nStopped by user.")

    # Print final statistics
    elapsed = time.time() - stats["start_time"]
    print("\n" + "=" * 70)
    print("Session Statistics")
    print("=" * 70)
    print(f"\nDuration: {elapsed:.1f} seconds")
    print(f"Frames Processed: {stats['frames_processed']}")
    print(f"Average FPS: {stats['frames_processed'] / elapsed:.1f}")

    if stats["field_counts"]:
        print(f"\nFields Received:")
        for field, count in sorted(stats["field_counts"].items()):
            print(f"  {field}: {count} frames")

    if stats["detections_per_frame"]:
        total_detections = sum(stats["detections_per_frame"])
        avg_detections = total_detections / len(stats["detections_per_frame"])
        max_detections = max(stats["detections_per_frame"])
        print(f"\nDetection Statistics:")
        print(f"  Total Detections: {total_detections}")
        print(f"  Average per Frame: {avg_detections:.1f}")
        print(f"  Max in Single Frame: {max_detections}")

    print("\n" + "=" * 70)
    print("\n💡 Benefits of DATA_ONLY mode:")
    print("  ✓ Significantly reduced bandwidth (no video sent back)")
    print("  ✓ Lower latency for data processing")
    print("  ✓ Ideal for headless/server deployments")
    print("  ✓ Perfect for analytics and logging use cases")
    print("\n")


if __name__ == "__main__":
    main()
