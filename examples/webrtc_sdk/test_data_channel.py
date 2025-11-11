"""Automated test for WebRTC data channel video streaming.

Success criteria:
1. 100% frame delivery (all frames processed)
2. Processing rate >= 5 fps
3. No errors in processing
4. Frames processed in order
"""

import asyncio
import base64
import cv2
import json
import requests
import sys
import time
from aiortc import RTCPeerConnection, RTCSessionDescription


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total, fps


async def run_test(video_path: str, max_frames: int = None):
    """Run data channel streaming test.
    
    Returns: (success: bool, stats: dict)
    """
    total_frames_in_video, video_fps = get_video_info(video_path)
    frames_to_process = min(max_frames or total_frames_in_video, total_frames_in_video)
    
    print(f"📹 Video: {frames_to_process}/{total_frames_in_video} frames @ {video_fps:.1f} FPS")
    
    start_time = time.time()
    pc = RTCPeerConnection()
    
    # Stats tracking
    stats = {
        "frames_sent": 0,
        "responses_received": 0,
        "frame_ids_received": [],
        "errors": [],
        "start_time": start_time,
    }
    
    # Create channels
    inference_ch = pc.createDataChannel("inference")
    upstream_ch = pc.createDataChannel("upstream_frames")
    
    @inference_ch.on("message")
    def on_response(msg):
        try:
            data = json.loads(msg)
            frame_id = data.get('video_metadata', {}).get('frame_id')
            stats["responses_received"] += 1
            stats["frame_ids_received"].append(frame_id)
            
            if data.get('errors'):
                stats["errors"].extend(data['errors'])
            
            if stats["responses_received"] % 50 == 0:
                elapsed = time.time() - start_time
                fps = stats["responses_received"] / elapsed
                print(f"  📊 {stats['responses_received']}/{frames_to_process} @ {fps:.1f} fps")
        except Exception as e:
            stats["errors"].append(str(e))
    
    @upstream_ch.on("open")
    def send_frames():
        print("📤 Sending frames...")
        cap = cv2.VideoCapture(video_path)
        
        for frame_id in range(1, frames_to_process + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            msg = json.dumps({
                "type": "frame",
                "frame_id": frame_id,
                "image": base64.b64encode(buf).decode()
            })
            upstream_ch.send(msg)
            stats["frames_sent"] = frame_id
        
        cap.release()
        print(f"✅ Sent {stats['frames_sent']} frames")
    
    # Establish WebRTC connection
    print("🔗 Establishing WebRTC connection...")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)
    
    resp = requests.post("http://localhost:9001/initialise_webrtc_worker", json={
        "api_key": "LKgvRJqgdbCml2ONofEx",
        "workflow_configuration": {
            "type": "WorkflowConfiguration",
            "workflow_id": "custom-workflow-3",
            "workspace_name": "leandro-starter",
            "image_input_name": "image",
        },
        "webrtc_offer": {
            "type": pc.localDescription.type,
            "sdp": pc.localDescription.sdp
        },
        "output_mode": "data_only",
        "use_data_channel_frames": True,
        "data_output": None,
    })
    
    if resp.status_code != 200:
        print(f"❌ Server error: {resp.status_code} - {resp.text}")
        return False, stats
    
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=resp.json()["sdp"], type=resp.json()["type"])
    )
    
    print("⏳ Waiting for processing to complete...")
    
    # Wait for all responses (with timeout)
    timeout_sec = 300
    for i in range(timeout_sec):
        await asyncio.sleep(1)
        if stats["responses_received"] >= frames_to_process:
            break
        if i > 0 and i % 30 == 0:
            print(f"  ⏱️  {i}s elapsed, {stats['responses_received']}/{frames_to_process} received")
    
    stats["end_time"] = time.time()
    stats["duration"] = stats["end_time"] - stats["start_time"]
    
    await asyncio.sleep(1)
    await pc.close()
    
    # Validate results
    success = validate_results(stats, frames_to_process)
    return success, stats


def validate_results(stats, expected_frames):
    """Validate test results against success criteria."""
    print("\n" + "="*50)
    print("📊 TEST RESULTS")
    print("="*50)
    
    print(f"Frames sent: {stats['frames_sent']}")
    print(f"Responses received: {stats['responses_received']}/{expected_frames}")
    print(f"Duration: {stats['duration']:.1f}s")
    
    fps = stats['responses_received'] / stats['duration'] if stats['duration'] > 0 else 0
    print(f"Processing rate: {fps:.1f} fps")
    
    # Check success criteria
    checks = []
    
    # 1. 100% frame delivery
    completeness = stats['responses_received'] / expected_frames * 100
    check1 = stats['responses_received'] == expected_frames
    checks.append(check1)
    print(f"\n✅ Criterion 1: 100% delivery - {completeness:.1f}% {'PASS' if check1 else 'FAIL'}")
    
    # 2. Processing rate >= 5 fps
    check2 = fps >= 5.0
    checks.append(check2)
    print(f"{'✅' if check2 else '❌'} Criterion 2: Speed >= 5fps - {fps:.1f} fps {'PASS' if check2 else 'FAIL'}")
    
    # 3. No errors
    check3 = len(stats['errors']) == 0
    checks.append(check3)
    if stats['errors']:
        print(f"❌ Criterion 3: No errors - {len(stats['errors'])} errors: {stats['errors'][:3]}")
    else:
        print(f"✅ Criterion 3: No errors - PASS")
    
    # 4. Frames in order
    ordered = stats['frame_ids_received'] == sorted(stats['frame_ids_received'])
    check4 = ordered
    checks.append(check4)
    print(f"{'✅' if check4 else '❌'} Criterion 4: Ordered delivery - {'PASS' if check4 else 'FAIL'}")
    
    print("\n" + "="*50)
    all_passed = all(checks)
    print(f"{'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("="*50 + "\n")
    
    return all_passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test WebRTC data channel video streaming")
    parser.add_argument("--video-path", default="/Users/balthasar/Downloads/times_square_2025-08-10_07-02-07.mp4")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to test (default: 100)")
    args = parser.parse_args()
    
    print("🧪 WebRTC Data Channel Video Streaming Test\n")
    
    success, stats = asyncio.run(run_test(args.video_path, args.max_frames))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

