import numpy as np
import imageio
from pathlib import Path
from IPython.display import display, Video, Image


def frames_to_mp4(
    frames: list[np.ndarray],
    path: str | Path = "animation.mp4",
    fps: int = 30,
    codec: str = "libx264",
    **writer_kwargs,
) -> Video:
    """
    Convert a list of RGBA uint8 NumPy frames (H, W, 4) to an MP4 and
    return an IPython.display.Video object for inline playback.
    """

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # NB: use imageio.get_writer (classic API)
    with imageio.get_writer(
        path,
        fps=fps,
        codec=codec,  # requires the imageio-ffmpeg plugin (pip install imageio[ffmpeg])
        format="mp4",  # forces FFmpeg writer; omit if you prefer auto-detect
        pixelformat="yuv420p",  # wide compatibility
        **writer_kwargs,
    ) as writer:
        for idx, f in enumerate(frames):
            if f.ndim != 3 or f.shape[-1] != 4:
                raise ValueError(f"Frame {idx} has shape {f.shape}, expected (H, W, 4)")
            writer.append_data(
                f[..., :3].astype(np.uint8)
            )  # drop alpha (MP4 can't store it)

    return path


def mp4_to_gif(
    path: str | Path = "animation.mp4",
    output_path: str | Path = "animation.gif",
    fps: int = 30,
):
    path = Path(path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all frames from the MP4 file and get the fps
    frames = []
    reader = imageio.get_reader(path)
    for frame in reader:
        frames.append(frame)
    reader.close()

    imageio.mimsave(
        output_path,
        frames,
        format="GIF",
        duration=1000 / fps,  # Default assuming 30fps from the input MP4
        loop=0,  # 0 means loop indefinitely
    )

    return output_path


def display_video(path: str | Path = "animation.mp4"):
    return display(
        Video(str(path), embed=True, html_attributes="controls loop autoplay")
    )


def display_image(path: str | Path = "animation.gif"):
    return display(Image(str(path)))
