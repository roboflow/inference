# Delayed action-recognition video

Use `workflow.json` with a persistent video workflow / InferencePipeline, passing
`model_id` for a fine-tuned sliding-window action-recognition model. Configure the
video sink or WebRTC `stream_output` to use the workflow output named `image`.
The model must receive the original input, and the visualizer must receive the
same stream's **latest** timeline. This example uses the existing Action Recognition Model v1's `timeline` output.

The example classifies every 5 seconds (`stride_seconds`). The model determines
the analysis window length independently. The visualizer displays images about
10 seconds earlier using `ceil(delay_seconds * source_fps)` frame indices. At
30 FPS, processing frame 601 displays frame 301 and the actions whose inclusive
start/end frame indices contain 301. Stride, window length and display delay are
three different settings. Ten seconds is a margin, not a guarantee that analysis
has completed or that later overlapping windows will not add evidence.

## Outputs

- `image`: a presentation image, downscaled before storage to a maximum side of
  640 pixels without upscaling or changing aspect ratio. Models should keep using
  the original image. During warmup this is a blank waiting screen.
- `displayed_frame_number`: the source index of the image actually displayed;
  null during warmup. Image video metadata also describes this old frame.
- `actions`: unique, sorted classes currently known for that displayed frame.
  No matching action does **not** prove that analysis completed without a match.
- `status`: `buffering` or `ready`. Ready describes image availability only.
- `model_error`: the model's error output, independent of display readiness.

## Runtime and memory

Use sequential frames with a stable video identifier and positive declared source
FPS. The visualizer pins the initial source FPS like the model. It never uses
processing/measured FPS or sleeps. Missing source FPS raises a clear error.
Dropped frames select the newest available predecessor of the target, so the
actual delay may grow. With irregular frame numbering this is not an exact delay
in seconds. WebRTC transport frame IDs/PTS can still describe the current request;
consumers should use `displayed_frame_number` for the content's identity.

The default memory budget is 256 MiB of retained pixels **across all streams** in
one block instance; rendering temporaries, input pixels and Python objects are
additional. A full 10-second delay at 30 FPS and 640x360 reserves about 199 MiB.
At 1920x1080 it needs about 1.75 GiB. Insufficient capacity raises an error before
warming the stream, rather than silently reducing its delay. Reduce
`max_frame_side`, reduce `delay_seconds`, or increase `max_buffer_mb`. Images are
owned CPU copies; the buffer retains no GPU tensor references. Up to 16 streams
are retained. Ended streams remain reserved until workflow restart; reconnects
with a lower frame number reset that stream. Delay, resolution configuration or
input dimension changes also restart its buffer. Duplicate frame calls do not
add a second buffered image.

This version targets continuous video. It does not flush the last delayed frames
of a finite clip or trigger an extra model analysis at end of stream. Stateful
HTTP requests routed to fresh instances cannot maintain the buffer. Synchronous
model latency still affects playback; this block does not make inference
asynchronous or guarantee real-time throughput.
