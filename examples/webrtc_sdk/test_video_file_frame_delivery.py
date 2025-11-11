"""
Test script to measure frame delivery completeness for video files.

Compares:
1. Media track + realtime_processing=True (may drop frames)
2. Media track + realtime_processing=False (should buffer all)

Usage:
  python examples/webrtc_sdk/test_video_file_frame_delivery.py \\
      --video-path ~/Downloads/times_square_2025-08-10_07-02-07.mp4 \\
      --workspace-name leandro-starter \\
      --workflow-id custom-workflow-3 \\
      --api-url http://localhost:9001 \\
      --api-key LKgvRJqgdbCml2ONofEx \\
      --stream-output image \\
      --data-output predictions
"""
import argparse
import os
import time
from pathlib import Path

import cv2

from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig


def get_video_metadata(video_path: str) -> dict:
    """Extract metadata from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }


def run_test(
    video_path: str,
    client: InferenceHTTPClient,
    workspace: str,
    workflow: str,
    image_input: str,
    stream_output: str | None,
    data_output: str | None,
    realtime_processing: bool,
) -> dict:
    """Run a single test configuration and collect metrics."""
    
    print(f"\n{'='*80}")
    print(f"Testing: realtime_processing={realtime_processing}")
    print(f"{'='*80}\n")
    
    # Prepare source and config
    source = VideoFileSource(video_path)
    stream_outputs = [stream_output] if stream_output else []
    data_outputs = [data_output] if data_output else []
    config = StreamConfig(
        stream_output=stream_outputs,
        data_output=data_outputs,
        realtime_processing=realtime_processing,
    )
    
    # Metrics
    metrics = {
        "realtime_processing": realtime_processing,
        "video_frames_received": 0,
        "data_messages_received": 0,
        "unique_frame_ids": set(),
        "start_time": None,
        "end_time": None,
        "errors": [],
    }
    
    try:
        with client.webrtc.stream(
            source=source,
            workflow=workflow,
            workspace=workspace,
            image_input=image_input,
            config=config,
        ) as session:
            
            # Register data handler
            @session.data.on_data()
            def on_message(msg):
                metrics["data_messages_received"] += 1
                try:
                    import json
                    parsed = json.loads(msg)
                    if "video_metadata" in parsed and "frame_id" in parsed["video_metadata"]:
                        frame_id = parsed["video_metadata"]["frame_id"]
                        metrics["unique_frame_ids"].add(frame_id)
                except Exception as e:
                    metrics["errors"].append(f"Failed to parse data message: {e}")
            
            metrics["start_time"] = time.time()
            
            # Consume video stream
            if stream_output:
                for frame in session.video():
                    metrics["video_frames_received"] += 1
                    # Don't display, just count
            else:
                # No video output, just wait for data
                session.wait(timeout=120)
            
            metrics["end_time"] = time.time()
    
    except Exception as e:
        metrics["errors"].append(f"Session error: {e}")
        if metrics["start_time"] and not metrics["end_time"]:
            metrics["end_time"] = time.time()
    
    return metrics


def print_results(video_meta: dict, results: list[dict]) -> None:
    """Print comparison of results."""
    print("\n" + "="*80)
    print("VIDEO FILE METADATA")
    print("="*80)
    print(f"Path: {video_meta['path']}")
    print(f"Total frames: {video_meta['total_frames']}")
    print(f"FPS: {video_meta['fps']:.2f}")
    print(f"Resolution: {video_meta['width']}x{video_meta['height']}")
    print(f"Duration: {video_meta['duration']:.2f}s")
    
    print("\n" + "="*80)
    print("TEST RESULTS COMPARISON")
    print("="*80)
    
    for metrics in results:
        elapsed = metrics["end_time"] - metrics["start_time"] if metrics["end_time"] and metrics["start_time"] else 0
        unique_frames = len(metrics["unique_frame_ids"])
        
        print(f"\nConfiguration: realtime_processing={metrics['realtime_processing']}")
        print("-" * 80)
        print(f"  Video frames received:     {metrics['video_frames_received']}")
        print(f"  Data messages received:    {metrics['data_messages_received']}")
        print(f"  Unique frame IDs:          {unique_frames}")
        print(f"  Processing time:           {elapsed:.2f}s")
        print(f"  Processing FPS:            {unique_frames / elapsed if elapsed > 0 else 0:.2f}")
        
        # Calculate completeness
        total_frames = video_meta['total_frames']
        if total_frames > 0:
            video_completeness = (metrics['video_frames_received'] / total_frames) * 100
            data_completeness = (unique_frames / total_frames) * 100
            print(f"  Video stream completeness: {video_completeness:.1f}%")
            print(f"  Data channel completeness: {data_completeness:.1f}%")
            print(f"  Frames dropped:            {total_frames - unique_frames}")
        
        if metrics["errors"]:
            print(f"  Errors: {len(metrics['errors'])}")
            for err in metrics["errors"][:3]:  # Show first 3
                print(f"    - {err}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if len(results) == 2:
        rt_true = next(r for r in results if r["realtime_processing"])
        rt_false = next(r for r in results if not r["realtime_processing"])
        
        rt_true_frames = len(rt_true["unique_frame_ids"])
        rt_false_frames = len(rt_false["unique_frame_ids"])
        total = video_meta['total_frames']
        
        print(f"Source video total frames:           {total}")
        print(f"Realtime=True processed:             {rt_true_frames} ({rt_true_frames/total*100:.1f}%)")
        print(f"Realtime=False processed:            {rt_false_frames} ({rt_false_frames/total*100:.1f}%)")
        print(f"Difference:                          {abs(rt_false_frames - rt_true_frames)} frames")
        
        if rt_false_frames < total:
            print(f"\n⚠️  Even with buffering, {total - rt_false_frames} frames were not processed!")
            print("    This suggests frames are being dropped somewhere else (encoding, network, etc.)")
        elif rt_false_frames == total:
            print(f"\n✅ With buffering, all {total} frames were processed!")
        
        if rt_true_frames < rt_false_frames:
            print(f"\n📉 Realtime mode dropped {rt_false_frames - rt_true_frames} frames compared to buffered mode")
        elif rt_true_frames == rt_false_frames == total:
            print(f"\n✅ No frames dropped in either mode!")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Video File Frame Delivery Test")
    p.add_argument(
        "--video-path", required=True, help="Path to video file to test"
    )
    p.add_argument("--api-url", default="http://localhost:9001")
    p.add_argument("--workspace-name", required=True)
    p.add_argument("--workflow-id", required=True)
    p.add_argument("--image-input-name", default="image")
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--stream-output",
        default=None,
        help="Name of the workflow output to stream (e.g., 'image')",
    )
    p.add_argument(
        "--data-output",
        default=None,
        help="Name of the workflow output to receive via data channel (e.g., 'predictions')",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Expand user path
    video_path = os.path.expanduser(args.video_path)
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    # Get video metadata
    print("Reading video metadata...")
    video_meta = get_video_metadata(video_path)
    video_meta["path"] = video_path
    
    # Initialize client
    client = InferenceHTTPClient.init(api_url=args.api_url, api_key=args.api_key)
    
    # Run tests
    results = []
    
    # Test 1: realtime_processing=True
    metrics_true = run_test(
        video_path=video_path,
        client=client,
        workspace=args.workspace_name,
        workflow=args.workflow_id,
        image_input=args.image_input_name,
        stream_output=args.stream_output,
        data_output=args.data_output,
        realtime_processing=True,
    )
    results.append(metrics_true)
    
    # Wait a bit between tests
    print("\nWaiting 3 seconds before next test...")
    time.sleep(3)
    
    # Test 2: realtime_processing=False
    metrics_false = run_test(
        video_path=video_path,
        client=client,
        workspace=args.workspace_name,
        workflow=args.workflow_id,
        image_input=args.image_input_name,
        stream_output=args.stream_output,
        data_output=args.data_output,
        realtime_processing=False,
    )
    results.append(metrics_false)
    
    # Print comparison
    print_results(video_meta, results)


if __name__ == "__main__":
    main()

