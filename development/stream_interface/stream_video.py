import argparse
import os
import subprocess
from glob import glob
from threading import Thread

BASE_STREAM_URL = "rtsp://localhost:8554/live"


def main(video_dir: str, n: int, id_offset: int) -> None:
    video_paths = glob(os.path.join(video_dir, "*.mp4")) + glob(os.path.join(video_dir, "*.webm"))
    print(video_paths)
    while len(video_paths) < n:
        video_paths = video_paths * 2
    video_paths = video_paths[:n]
    threads = []
    for idx, video_path in enumerate(video_paths):
        stream_url = f"{BASE_STREAM_URL}{id_offset + idx}.stream"
        print(f"Streaming {video_path} under {stream_url}")
        threads.append(
            stream_video(video_path=video_path, stream_url=stream_url)
        )
    for t in threads:
        t.join()


def stream_video(video_path: str, stream_url: str) -> Thread:
    return run_command_in_thread(
        command=f"ffmpeg -re -stream_loop -1 -i ".split()
        + [f"{video_path}"]
        + f"-f rtsp -rtsp_transport tcp {stream_url}".split()
    )


def run_command_in_thread(command: list) -> Thread:
    thread = Thread(target=run_command, args=(command,))
    thread.start()
    return thread


def run_command(command: list) -> int:
    completed = subprocess.run(command)
    return completed.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to emit RTSP streams to RTSP server")
    parser.add_argument(
        "--video_dir", type=str, help="Directory with videos", required=True
    )
    parser.add_argument(
        "--n", type=int, help="Number of streams", required=False, default=6
    )
    parser.add_argument(
        "--id_offset", type=int, help="Offset of stream_id", required=False, default=0,
    )
    args = parser.parse_args()
    main(video_dir=args.video_dir, n=args.n, id_offset=args.id_offset)
