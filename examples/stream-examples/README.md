# Stream Examples

This folder contains examples that show how to use the `inference.Stream()` method.

## Examples

- `video.py`: Run an object detection model hosted on Roboflow on frames in an `.mp4` video.
- `webcam.py`: Run an object detection model hosted on Roboflow on frames from a webcam.
- `rtsp.py`: Run an object detection model hosted on Roboflow on frames from an RTSP stream.
- `track.py`: Run an object detection model with ByteTrack on frames from a webcam. This example both identifies and tracks objects in a stream.
- `paintwtf.py`: Use CLIP to identify the similarity between a video frame and a text prompt. Uses a webcam. This project is inspired by [paint.wtf](https://paint.wtf).