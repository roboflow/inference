"""Example streaming video frames via WebRTC data channel."""

import argparse
import cv2
import time

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import (
    DataChannelVideoSource,
    FrameTransport,
    StreamConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("WebRTC data channel video example")
    parser.add_argument("--video-path", required=True, help="Path to video file to stream")
    parser.add_argument("--api-url", default="http://localhost:9001")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument(
        "--image-input-name",
        default="image",
        help="Name of the image input in the workflow",
    )
    parser.add_argument(
        "--output-mode",
        choices=["data_only", "video_only", "both"],
        default="both",
        help="What outputs to request from the server",
    )
    parser.add_argument(
        "--stream-output",
        default=None,
        help="Workflow output to stream back as video (auto-detected if omitted)",
    )
    parser.add_argument(
        "--data-output",
        default=None,
        help="Comma separated list of workflow outputs to receive via data channel. Use 'all' (default) or 'none'.",
    )
    return parser.parse_args()


def build_stream_config(args: argparse.Namespace) -> StreamConfig:
    data_output = None
    if args.data_output:
        token = args.data_output.strip().lower()
        if token == "none":
            data_output = []
        elif token == "all":
            data_output = None
        else:
            data_output = [item.strip() for item in args.data_output.split(",") if item.strip()]

    return StreamConfig(
        stream_output=[args.stream_output] if args.stream_output else [],
        data_output=data_output,
        frame_transport=FrameTransport.DATA_CHANNEL,
        output_mode=args.output_mode,
    )


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"total_frames": 0, "fps": 0}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return {"total_frames": total_frames, "fps": fps}


def main() -> None:
    print("Starting data channel video streaming...")
    args = parse_args()
    
    # Get video info
    video_info = get_video_info(args.video_path)
    total_frames = video_info["total_frames"]
    fps = video_info["fps"]
    
    print(f"Video path: {args.video_path}")
    print(f"Video info: {total_frames} frames @ {fps:.2f} FPS")
    print(f"Workspace: {args.workspace_name}")
    print(f"Workflow: {args.workflow_id}")
    print(f"Output mode: {args.output_mode}")
    
    client = InferenceHTTPClient.init(api_url=args.api_url, api_key=args.api_key)
    print("HTTP client initialized")

    source = DataChannelVideoSource(args.video_path)
    print("Video source created")
    
    config = build_stream_config(args)
    print(f"Stream config: transport={config.frame_transport}, mode={config.output_mode}")

    print("Opening WebRTC session...")
    
    # Track messages received
    message_count = [0]
    start_time = time.time()
    last_report = [start_time]
    
    with client.webrtc.stream(
        source=source,
        workflow=args.workflow_id,
        workspace=args.workspace_name,
        image_input=args.image_input_name,
        config=config,
    ) as session:
        print("WebRTC session opened!")
        
        # Print any structured data returned via inference data channel
        @session.data.on_data()
        def on_message(msg):  # noqa: ANN001
            message_count[0] += 1
            current_time = time.time()
            
            # Report progress every 30 messages or every 2 seconds
            if message_count[0] % 30 == 0 or (current_time - last_report[0]) > 2.0:
                elapsed = current_time - start_time
                fps_received = message_count[0] / elapsed if elapsed > 0 else 0
                progress = (message_count[0] / total_frames * 100) if total_frames > 0 else 0
                print(f"[{elapsed:.1f}s] Received {message_count[0]}/{total_frames} messages ({progress:.1f}%) @ {fps_received:.1f} msg/s")
                last_report[0] = current_time
            
            # Send EOF when all frames are processed
            if message_count[0] >= total_frames and hasattr(source, 'send_eof_when_ready'):
                source.send_eof_when_ready(message_count[0])

        print(f"Waiting for data (mode={config.output_mode})...")
        
        # Monitor sending progress
        last_send_report = [start_time]
        
        def report_send_progress():
            if hasattr(source, 'get_stats'):
                stats = source.get_stats()
                elapsed = time.time() - start_time
                print(f"  SENDING: {stats['frames_sent']}/{total_frames} frames ({stats['frames_sent']/total_frames*100:.1f}%), {stats['chunks_sent']} chunks @ {stats['frames_sent']/elapsed:.1f} fps")
        
        # If output_mode includes video, show frames returned by server
        if config.output_mode in ("video_only", "both"):
            print("Expecting video frames...")
            frame_count = 0
            for frame in session.video():
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_received = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Received {frame_count} video frames @ {fps_received:.1f} FPS")
                    report_send_progress()
                cv2.imshow("WebRTC Data Channel", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            print(f"Total video frames received: {frame_count}")
            report_send_progress()
        else:
            # Otherwise just block until processing completes with periodic updates
            print("Data-only mode, waiting for completion...")
            
            # Periodically report sending progress
            # Wait with a generous timeout (10 minutes)
            try:
                session.wait(timeout=600)
            except KeyboardInterrupt:
                print("\nInterrupted by user")
            
            # Final progress report
            time.sleep(1)
            report_send_progress()
            
            elapsed = time.time() - start_time
            fps_received = message_count[0] / elapsed if elapsed > 0 else 0
            print(f"\n=== FINAL STATS ===")
            report_send_progress()
            print(f"Total messages received: {message_count[0]}/{total_frames}")
            print(f"Completion: {message_count[0] / total_frames * 100:.1f}%")
            print(f"Time elapsed: {elapsed:.1f}s")
            print(f"Average rate: {fps_received:.1f} msg/s")
            print(f"Video FPS: {fps:.1f}")

    print("Closing session...")
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main()
