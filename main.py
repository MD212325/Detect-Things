import cv2
import numpy as np
import argparse
import sys
import os
import urllib.request

# Default model URLs
DEFAULT_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/MediosZ/MobileNet-SSD/master/"
    "mobilenet/MobileNetSSD_deploy.prototxt"
)
DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/MediosZ/MobileNet-SSD/master/"
    "mobilenet/MobileNetSSD_deploy.caffemodel"
)

# Default local filenames
DEFAULT_PROTOTXT = "MobileNetSSD_deploy.prototxt"
DEFAULT_MODEL = "MobileNetSSD_deploy.caffemodel"


def download_file(url: str, dest: str):
    try:
        print(f"[INFO] Downloading {url} to {dest}...")
        urllib.request.urlretrieve(url, dest)
        print(f"[INFO] Download complete.")
    except Exception as e:
        print(f"ERROR: Failed to download {url}. {e}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Object Detection via Video Source')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', type=str,
                        help='DroidCam video stream URL (e.g., http://192.168.1.10:4747/video)')
    group.add_argument('--device', type=int,
                        help='Local webcam device index (e.g., 0 for default)')
    parser.add_argument('--prototxt', type=str,
                        help='Local path or URL to MobileNet SSD .prototxt file')
    parser.add_argument('--model', type=str,
                        help='Local path or URL to MobileNet SSD .caffemodel file')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum confidence threshold for detections')
    parser.add_argument('--classes', type=str, default='all',
                        help='Comma-separated list of classes to detect (e.g., person,car) or "all"')
    return parser.parse_args()


def prepare_model_paths(pt_arg, model_arg):
    # Determine prototxt source and destination
    if pt_arg and pt_arg.startswith('http'):
        prototxt_url = pt_arg
        prototxt_path = DEFAULT_PROTOTXT
    elif pt_arg:
        prototxt_url = None
        prototxt_path = pt_arg
    else:
        prototxt_url = DEFAULT_PROTOTXT_URL
        prototxt_path = DEFAULT_PROTOTXT

    # Determine model source and destination
    if model_arg and model_arg.startswith('http'):
        model_url = model_arg
        model_path = DEFAULT_MODEL
    elif model_arg:
        model_url = None
        model_path = model_arg
    else:
        model_url = DEFAULT_MODEL_URL
        model_path = DEFAULT_MODEL

    # Download if URL provided
    if prototxt_url and not os.path.isfile(prototxt_path):
        download_file(prototxt_url, prototxt_path)
    if model_url and not os.path.isfile(model_path):
        download_file(model_url, model_path)

    return prototxt_path, model_path


def main():
    args = parse_args()

    # Prepare model files
    prototxt_path, model_path = prepare_model_paths(args.prototxt, args.model)

    # Load class labels
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    # Parse desired classes
    if args.classes.lower() == 'all':
        target_classes = None
    else:
        target_classes = set(cls.strip() for cls in args.classes.split(','))

    # Validate target classes
    if target_classes:
        invalid = target_classes - set(CLASSES)
        if invalid:
            print(f"ERROR: Invalid class names: {', '.join(invalid)}")
            sys.exit(1)

    # Load model
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    except Exception as e:
        print(f"ERROR: Could not load model files. {e}")
        sys.exit(1)

    # Initialize video capture
    if args.device is not None:
        cap = cv2.VideoCapture(args.device)
        print(f"[INFO] Using local webcam device {args.device}.")
    else:
        cap = cv2.VideoCapture(args.source, cv2.CAP_ANY)
        print(f"[INFO] Connecting to video source {args.source}.")

    if not cap.isOpened():
        print("ERROR: Cannot open video source.")
        sys.exit(1)

    print("[INFO] Starting detection. Press 'q' to quit.")

    while True:
        try:
            ret, frame = cap.read()
        except cv2.error as e:
            print(f"ERROR: Failed to read frame ({e}). Exiting.")
            break

        if not ret or frame is None:
            print("[WARN] Empty frame; check your connection.")
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < args.confidence:
                continue

            idx = int(detections[0, 0, i, 1])
            label_text = CLASSES[idx]

            # Skip if not in target
            if target_classes and label_text not in target_classes:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{label_text}: {confidence:.2f}"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
