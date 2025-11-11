"""Simple data channel video streaming test - minimal implementation."""

import argparse
import asyncio
import base64
import cv2
import json
import logging
import requests
from threading import Thread
from aiortc import RTCPeerConnection, RTCSessionDescription

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-url", default="http://localhost:9001")
    parser.add_argument("--output-mode", default="data_only", choices=["data_only", "both"])
    return parser.parse_args()


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps


async def run_streaming(args):
    total_frames, video_fps = get_video_info(args.video_path)
    print(f"Video: {total_frames} frames @ {video_fps:.2f} FPS", flush=True)
    
    # Setup
    print("Creating peer connection...", flush=True)
    loop = asyncio.get_event_loop()
    pc = RTCPeerConnection()
    
    # Track stats
    frames_sent = [0]
    responses_received = [0]
    
    # Create data channels
    print("Creating data channels...", flush=True)
    inference_channel = pc.createDataChannel("inference")
    upstream_channel = pc.createDataChannel("upstream_frames")
    print("Data channels created", flush=True)
    
    @inference_channel.on("message")
    def on_inference_message(msg):
        responses_received[0] += 1
        if responses_received[0] % 30 == 0 or responses_received[0] == total_frames:
            data = json.loads(msg)
            frame_id = data.get('video_metadata', {}).get('frame_id', '?')
            print(f"  Received {responses_received[0]}/{total_frames} responses (frame_id={frame_id})")
    
    @upstream_channel.on("open")
    def send_frames():
        print(f"Upstream channel open, sending {total_frames} frames...")
        
        cap = cv2.VideoCapture(args.video_path)
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode('ascii')
            
            message = json.dumps({
                "type": "frame",
                "frame_id": frame_id,
                "image": image_b64
            })
            
            upstream_channel.send(message)
            frames_sent[0] = frame_id
            
            if frame_id % 100 == 0:
                print(f"  Sent {frame_id}/{total_frames} frames")
        
        cap.release()
        print(f"✓ Sent all {frames_sent[0]} frames")
    
    # Create offer
    print("Creating offer...", flush=True)
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    print("Offer created, waiting for ICE...", flush=True)
    
    # Wait for ICE gathering
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)
    
    print("ICE complete, sending to server...", flush=True)
    
    # Send to server
    payload = {
        "api_key": args.api_key,
        "workflow_configuration": {
            "type": "WorkflowConfiguration",
            "workflow_id": args.workflow,
            "workspace_name": args.workspace,
            "image_input_name": "image",
        },
        "webrtc_offer": {
            "type": pc.localDescription.type,
            "sdp": pc.localDescription.sdp,
        },
        "output_mode": args.output_mode,
        "use_data_channel_frames": True,
        "data_output": None,  # All outputs
    }
    
    print("Initializing WebRTC worker...", flush=True)
    resp = requests.post(f"{args.api_url}/initialise_webrtc_worker", json=payload)
    resp.raise_for_status()
    answer = resp.json()
    print("Got answer from server", flush=True)
    
    # Set remote description
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )
    
    print("WebRTC session established, waiting for processing...", flush=True)
    
    # Wait for all responses
    while responses_received[0] < total_frames:
        await asyncio.sleep(1)
        if responses_received[0] > 0 and responses_received[0] % 100 == 0:
            print(f"  Still waiting... {responses_received[0]}/{total_frames}")
    
    print(f"\n✅ SUCCESS!")
    print(f"  Frames sent: {frames_sent[0]}")
    print(f"  Responses received: {responses_received[0]}")
    print(f"  Completion: {responses_received[0]/total_frames*100:.1f}%")
    
    await pc.close()
    return responses_received[0] == total_frames


def main():
    args = parse_args()
    success = asyncio.run(run_streaming(args))
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

