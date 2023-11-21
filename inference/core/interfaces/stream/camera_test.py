import cv2


def main() -> None:
    stream = cv2.VideoCapture(0)
    while stream.isOpened():
        status, image = stream.read()
        if not status:
            break
        cv2.imshow("stream", image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    print("DONE")


if __name__ == "__main__":
    main()
