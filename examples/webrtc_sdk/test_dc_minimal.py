"""Minimal test - just 10 frames."""
import sys
print("Starting...", flush=True)

import asyncio
import base64
import cv2
import json
import requests
from aiortc import RTCPeerConnection, RTCSessionDescription

print("Imports done", flush=True)

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/balthasar/Downloads/times_square_2025-08-10_07-02-07.mp4"
MAX_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 100

async def test():
    print(f"Testing with {MAX_FRAMES} frames from {VIDEO_PATH}", flush=True)
    
    pc = RTCPeerConnection()
    received = [0]
    
    inference_ch = pc.createDataChannel("inference")
    upstream_ch = pc.createDataChannel("upstream_frames")
    
    @inference_ch.on("message")
    def on_msg(msg):
        received[0] += 1
        print(f"✓ Response {received[0]}", flush=True)
    
    @upstream_ch.on("open")
    def send():
        print("Channel open, sending frames...", flush=True)
        cap = cv2.VideoCapture(VIDEO_PATH)
        for i in range(1, MAX_FRAMES + 1):
            ret, frame = cap.read()
            if not ret:
                break
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            msg = json.dumps({"type": "frame", "frame_id": i, "image": base64.b64encode(buf).decode()})
            upstream_ch.send(msg)
            print(f"  Sent frame {i}", flush=True)
        cap.release()
        print(f"All {MAX_FRAMES} frames sent!", flush=True)
    
    print("Creating offer...", flush=True)
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)
    
    print("Posting to server...", flush=True)
    resp = requests.post("http://localhost:9001/initialise_webrtc_worker", json={
        "api_key": "LKgvRJqgdbCml2ONofEx",
        "workflow_configuration": {
            "type": "WorkflowConfiguration",
            "workflow_id": "custom-workflow-3",
            "workspace_name": "leandro-starter",
            "image_input_name": "image",
        },
        "webrtc_offer": {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp},
        "output_mode": "data_only",
        "use_data_channel_frames": True,
        "data_output": None,
    })
    
    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} - {resp.text}", flush=True)
        return False
    
    print(f"Server responded: {resp.status_code}", flush=True)
    answer = resp.json()
    
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    print("Session established!", flush=True)
    
    # Wait for responses (must keep connection alive!)
    timeout_seconds = 300  # 5 minutes
    for i in range(timeout_seconds):
        await asyncio.sleep(1)
        if received[0] >= MAX_FRAMES:
            print(f"All responses received after {i}s!", flush=True)
            break
        if i > 0 and i % 30 == 0:
            print(f"  Still waiting... {received[0]}/{MAX_FRAMES} after {i}s", flush=True)
    
    print(f"\nFINAL: {received[0]}/{MAX_FRAMES} responses", flush=True)
    
    # Keep connection alive a bit longer before closing
    await asyncio.sleep(2)
    await pc.close()
    return received[0] == MAX_FRAMES

if __name__ == "__main__":
    success = asyncio.run(test())
    print(f"{'✅ SUCCESS' if success else '❌ FAILED'}", flush=True)
    sys.exit(0 if success else 1)

