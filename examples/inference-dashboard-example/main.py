import cv2
import pandas as pd
import pickle
import requests
import matplotlib.pyplot as plt
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Process video and extract insights")
    parser.add_argument("--dataset_id", help="Dataset ID (required)")
    parser.add_argument("--version_id", default="1", help="Version ID (default: 1)")
    parser.add_argument("--api_key", help="API key (required)")
    parser.add_argument("--video_path", help="Path to the video (required)")
    parser.add_argument("--interval_minutes", type=int, default=1, help="Interval in seconds (default: 60)")
    return parser.parse_args()


def extract_frames(video_path, interval_minutes):
    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (fps * interval_minutes) == 0:
            frames.append(frame)
            timestamps.append(frame_count / fps)
        frame_count += 1
    cap.release()
    return frames, timestamps


def fetch_predictions(base_url, frames, timestamps, dataset_id, version_id, api_key, confidence=0.5):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    df_rows = []
    for idx, frame in enumerate(frames):
        numpy_data = pickle.dumps(frame)
        res = requests.post(
            f"{base_url}/{dataset_id}/{version_id}",
            data=numpy_data,
            headers=headers,
            params={"api_key": api_key, "confidence": confidence, "image_type": "numpy"}
        )
        predictions = res.json()

        for pred in predictions['predictions']:
            time_interval = f"{int(timestamps[idx] // 60)}:{int(timestamps[idx] % 60):02}"
            row = {
                "timestamp": time_interval,
                "time": predictions['time'],
                "x": pred["x"],
                "y": pred["y"],
                "width": pred["width"],
                "height": pred["height"],
                "pred_confidence": pred["confidence"],
                "class": pred["class"]
            }
            df_rows.append(row)

    df = pd.DataFrame(df_rows)
    df['seconds'] = df['timestamp'].str.split(':').apply(lambda x: int(x[0])*60 + int(x[1]))
    df = df.sort_values(by="seconds")
    return df

def plot_and_save(data, title, filename, ylabel, stacked=False, legend_title=None, legend_loc=None, legend_bbox=None):
    plt.style.use('dark_background')
    data.plot(kind='bar', stacked=stacked, figsize=(15,7))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Timestamp (in minutes:seconds)')
    
    if legend_title:
        plt.legend(title=legend_title, loc=legend_loc, bbox_to_anchor=legend_bbox)
    
    plt.tight_layout()
    plt.savefig(filename)

def main():
    args = parse_args()
    base_url = "http://localhost:9001"
    video_path = args.video_path
    dataset_id = args.dataset_id
    version_id = args.version_id
    api_key = args.api_key
    interval_minutes = args.interval_minutes * 60


    frames, timestamps = extract_frames(video_path, interval_minutes)
    df = fetch_predictions(base_url, frames, timestamps, dataset_id, version_id, api_key)

    if not os.path.exists("results"):
        os.makedirs("results")

    #saving predictions response to csv
    df.to_csv("results/predictions.csv", index=False)

    # Transform timestamps to minutes and group
    df['minutes'] = df['timestamp'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    object_counts_per_interval = df.groupby('minutes').size().sort_index()
    object_counts_per_interval.index = object_counts_per_interval.index.map(lambda x: f"{x // 60}:{x % 60:02}")
    object_counts_per_interval.to_csv("results/object_counts_per_interval.csv")

    # Quick insights
    print(f"Total unique objects detected: {df['class'].nunique()}")
    print(f"Most frequently detected object: {df['class'].value_counts().idxmax()}")
    print(f"Time interval with the most objects detected: {object_counts_per_interval.idxmax()}")
    print(f"Time interval with the least objects detected: {object_counts_per_interval.idxmin()}")

    plot_and_save(object_counts_per_interval, 'Number of Objects Detected Over Time', "results/objects_over_time_d.png", 'Number of Objects')

    # Group by timestamp and class, then sort by minutes
    objects_by_class_per_interval = df.groupby(['minutes', 'class']).size().unstack(fill_value=0).sort_index()
    objects_by_class_per_interval.index = objects_by_class_per_interval.index.map(lambda x: f"{x // 60}:{x % 60:02}")
    objects_by_class_per_interval.to_csv("results/object_counts_by_class_per_interval.csv")

    plot_and_save(objects_by_class_per_interval, 'Number of Objects Detected Over Time by Class', "results/objects_by_class_over_time.png", 'Number of Objects', True, "Object Class", "center left", (1, 0.5))


if __name__ == "__main__":
    main()
