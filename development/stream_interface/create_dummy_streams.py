import argparse
import os
import subprocess
import tempfile
from glob import glob
from threading import Thread
import yaml


CONFIG = {"protocols": ["tcp"], "paths": {"all": {"source": "publisher"}}}
BASE_STREAM_URL = "rtsp://localhost:8554/live"


def main(video_dir: str, n: int) -> None:
    video_paths = glob(os.path.join(video_dir, "*.mp4")) + glob(os.path.join(video_dir, "*.webm"))
    video_paths = video_paths[:n]
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "rtsp-simple-server.yml")
            with open(config_path, "w") as f:
                yaml.dump(CONFIG, f)
            run_server(config_path=config_path)
            threads = []
            for idx, video_path in enumerate(video_paths):
                stream_url = f"{BASE_STREAM_URL}{idx}.stream"
                print(f"Streaming {video_path} under {stream_url}")
                threads.append(
                    stream_video(video_path=video_path, stream_url=stream_url)
                )
            for t in threads:
                t.join()
    finally:
        kill_server()


def run_server(config_path: str) -> None:
    command = (
        f"docker run --rm --name rtsp_server -d -v {config_path}:/rtsp-simple-server.yml "
        f"-p 8554:8554 aler9/rtsp-simple-server:v1.3.0"
    )
    return_code = run_command(command=command.split())
    if return_code != 0:
        raise RuntimeError("Could not run RTSP server!")


def kill_server() -> None:
    run_command(command="docker kill rtsp_server".split())


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
    parser = argparse.ArgumentParser("Script to run run dummy RTSP streams")
    parser.add_argument(
        "--video_dir", type=str, help="Directory with videos", required=True
    )
    parser.add_argument(
        "--n", type=int, help="Number of streams", required=False, default=6
    )
    args = parser.parse_args()
    main(video_dir=args.video_dir, n=args.n)
