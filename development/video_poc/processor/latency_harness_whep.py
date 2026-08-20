"""WHEP-side probe for the latency harness: subscribes to a relay stream over
WebRTC (same path the browser preview uses) and decodes the pixel clock.
Splits 'publisher/relay adds latency' from 'RTSP serving/client adds latency'.
"""

import argparse
import asyncio
import sys
import time

import requests

from latency_harness import BARS, BAR_W, HEIGHT, decode_frame, now_ms


async def probe(whep_url: str, samples: int):
    from aiortc import RTCPeerConnection, RTCSessionDescription

    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    done = asyncio.Event()
    lat = []
    warmup = 30

    @pc.on("track")
    def on_track(track):
        async def reader():
            while len(lat) < samples + warmup:
                frame = await track.recv()
                image = frame.to_ndarray(format="bgr24")
                if image.shape[0] != HEIGHT:
                    # WHEP may deliver scaled frames; rescale bar geometry
                    import cv2

                    image = cv2.resize(image, (BARS * BAR_W, HEIGHT))
                value = decode_frame(image)
                delta = (now_ms() - value) & 0xFFFFFFFF
                if delta < 60_000:
                    lat.append(delta)
            done.set()

        asyncio.ensure_future(reader())

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)
    resp = requests.post(
        whep_url,
        data=pc.localDescription.sdp,
        headers={"Content-Type": "application/sdp"},
        timeout=10,
    )
    resp.raise_for_status()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=resp.text, type="answer"))
    try:
        await asyncio.wait_for(done.wait(), timeout=60)
    finally:
        await pc.close()
    usable = lat[warmup:]
    if not usable:
        print("[harness-whep] no valid samples")
        sys.exit(2)
    usable.sort()
    n = len(usable)
    print(
        f"[harness-whep] samples={n} p50={usable[n // 2]}ms "
        f"p90={usable[int(n * 0.9)]}ms min={usable[0]}ms max={usable[-1]}ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8889/lat-test/whep")
    parser.add_argument("--samples", type=int, default=120)
    args = parser.parse_args()
    asyncio.run(probe(args.url, args.samples))
