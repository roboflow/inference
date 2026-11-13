# Action recognition on video

`workflow.json` runs an Action Recognition Model on a video stream and draws
its results. Use it with a persistent video workflow (`InferencePipeline` or
WebRTC), pass `model_id` for a fine-tuned sliding-window model, and point the
video sink or WebRTC `stream_output` at the workflow output named `image`.

An action recognition result always describes frames the stream has already
passed. The model needs a window of video before it can answer, and the answer
takes time to compute. The example shows two ways to present that.

## Display modes

Set the `display_mode` input.

`live` (default)

- The video is the current frame. Nothing is delayed and no frame is stored.
- A strip at the bottom covers the last 30 seconds: one lane per class, a bar
  for each action range, and a hatched zone for frames not analyzed yet.
- A badge names the actions at the last analyzed frame and how long ago that
  was, for example `Punch` `4 s ago`.

`synchronized`

- The video plays late, so each label sits on the frame it describes.
- You do not set the delay. The visualizer follows `analyzed_until_frame` and
  plays behind the stream by about one stride plus one model call.
- Until the first result arrives, the first frame is held with an
  `Analyzing...` badge. Playback then continues from that frame.
- If results stop arriving, playback resumes after `max_delay_seconds`
  (default 60) with the status `behind`.

Use `live` to monitor a camera. Use `synchronized` to demo or record a result.

## Why the model runs in the background

The example sets `background_inference: true` on the model. A live stream then
hands each window to a worker thread, and the video does not wait for the
model. Without it, every classification blocks the stream for the length of
one model call, and a realtime source drops frames meanwhile.

Video files always classify inside the step, whatever this setting says. A
file is not paced by the clock, so a background worker would fall behind and
leave most windows unclassified. For a file, the `timeline` on the last frame
covers the whole clip.

## Outputs

- `image`: the drawn frame. In `synchronized` it is an earlier frame,
  downscaled to at most 1280 pixels on its longest side.
- `displayed_frame_number`: the source index of the frame in `image`.
- `actions`: the classes at `actions_frame_number`. That is the displayed
  frame in `synchronized` and the last analyzed frame in `live`. It is empty
  while no frame has a result.
- `status`: `analyzing` before the first result, then `live`, `synchronized`
  or `behind`.
- `analyzed_until_frame`: the last frame the model has a result for. A frame
  after it has no result yet; that is different from "no action".
- `model_error`: the model's error text for that frame. With background
  inference, an error appears on the frame that collects the failed call.

## Limits

- `synchronized` keeps compressed frames in host memory, within
  `max_buffer_mb` (default 256) across all streams of the step. When the
  budget is too small, the oldest frames are dropped, the delay shrinks and
  the status becomes `behind`. Lower `max_frame_side` or raise the budget.
- `synchronized` does not suit video files: the file ends before its delayed
  tail is shown.
- One classification runs per stream at a time. When the model is slower than
  the stride, the next call waits and takes the newest window. If that leaves
  frames no window covered, `model_error` names them.
- Both blocks need a persistent workflow. Stateless HTTP requests routed to
  fresh instances keep no stream state.
