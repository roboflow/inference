import base64
import os

import cv2
import numpy as np
import requests

IMG_PATH = "image.jpg"
ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]
DISTANCE_TO_OBJECT = 1000  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm
GAZE_DETECTION_URL = "http://127.0.0.1:9001/gaze/gaze_detection?api_key=" + ROBOFLOW_API_KEY


def detect_gazes(frame: np.ndarray):
    img_encode = cv2.imencode(".jpg", frame)[1]
    img_base64 = base64.b64encode(img_encode)
    resp = requests.post(
        GAZE_DETECTION_URL,
        json={
            "api_key": ROBOFLOW_API_KEY,
            "image": {"type": "base64", "value": img_base64.decode("utf-8")},
        },
    )
    # print(resp.json())
    gazes = resp.json()[0]["predictions"]
    return gazes


def draw_gaze(img: np.ndarray, gaze: dict):
    # draw face bounding box
    face = gaze["face"]
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    # draw gaze arrow
    _, imgW = img.shape[:2]
    arrow_length = imgW / 2
    dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
    dy = -arrow_length * np.sin(gaze["pitch"])
    cv2.arrowedLine(
        img,
        (int(face["x"]), int(face["y"])),
        (int(face["x"] + dx), int(face["y"] + dy)),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )

    # draw keypoints
    for keypoint in face["landmarks"]:
        color, thickness, radius = (0, 255, 0), 2, 2
        x, y = int(keypoint["x"]), int(keypoint["y"])
        cv2.circle(img, (x, y), thickness, color, radius)

    # draw label and score
    label = "yaw {:.2f}  pitch {:.2f}".format(
        gaze["yaw"] / np.pi * 180, gaze["pitch"] / np.pi * 180
    )
    cv2.putText(
        img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    return img


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        gazes = detect_gazes(frame)

        if len(gazes) == 0:
            continue

        # draw face & gaze
        gaze = gazes[0]
        draw_gaze(frame, gaze)

        image_height, image_width = frame.shape[:2]

        length_per_pixel = HEIGHT_OF_HUMAN_FACE / gaze["face"]["height"]

        dx = -DISTANCE_TO_OBJECT * np.tan(gaze["yaw"]) / length_per_pixel
        # 100000000 is used to denote out of bounds
        dx = dx if not np.isnan(dx) else 100000000
        dy = (
            -DISTANCE_TO_OBJECT
            * np.arccos(gaze["yaw"])
            * np.tan(gaze["pitch"])
            / length_per_pixel
        )
        dy = dy if not np.isnan(dy) else 100000000
        gaze_point = int(image_width / 2 + dx), int(image_height / 2 + dy)

        quadrants = [
            (
                "center",
                (
                    int(image_width / 4),
                    int(image_height / 4),
                    int(image_width / 4 * 3),
                    int(image_height / 4 * 3),
                ),
            ),
            ("top_left", (0, 0, int(image_width / 2), int(image_height / 2))),
            (
                "top_right",
                (int(image_width / 2), 0, image_width, int(image_height / 2)),
            ),
            (
                "bottom_left",
                (0, int(image_height / 2), int(image_width / 2), image_height),
            ),
            (
                "bottom_right",
                (
                    int(image_width / 2),
                    int(image_height / 2),
                    image_width,
                    image_height,
                ),
            ),
        ]

        for quadrant, (x_min, y_min, x_max, y_max) in quadrants:
            if x_min <= gaze_point[0] <= x_max and y_min <= gaze_point[1] <= y_max:
                print(quadrant)
                break

        cv2.circle(frame, gaze_point, 25, (0, 0, 255), -1)

        cv2.imshow("gaze", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
