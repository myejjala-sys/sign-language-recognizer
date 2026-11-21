import math
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"


def create_detector():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (approx. 8MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def extract_features(lm):
    """42 numbers: 21 points (x, y) relative to the wrist, scaled by hand size."""
    wrist = lm[0]
    coords = [(p.x - wrist.x, p.y - wrist.y) for p in lm]
    scale = max(math.hypot(x, y) for x, y in coords) or 1e-6
    feats = []
    for x, y in coords:
        feats += [x / scale, y / scale]
    return feats


def draw_points(frame, lm):
    h, w, _ = frame.shape
    for p in lm:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 255, 0), -1)
