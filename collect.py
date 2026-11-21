"""Collect training data.
Show a sign, then PRESS the matching letter key ONCE.
It automatically records SAMPLES_PER_PRESS samples to data.csv.
Move your hand a little (closer, farther, slight tilt) while it records.
Press the same letter again to add more. Press ESC to quit."""
import csv
import time
from collections import Counter

import cv2
import mediapipe as mp

from features import create_detector, draw_points, extract_features, open_camera

# Only these letters will be recorded. Change to the letters you want.
ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXY"
SAMPLES_PER_PRESS = 100

detector = create_detector()
cap = open_camera()
counts = Counter()
start = time.time()
recording = None   # letter being recorded
remaining = 0      # samples left for this press

with open("data.csv", "a", newline="") as f:
    writer = csv.writer(f)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(img, int((time.time() - start) * 1000))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if remaining == 0 and 97 <= key <= 122 and chr(key).upper() in ALLOWED:
            recording = chr(key).upper()
            remaining = SAMPLES_PER_PRESS

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_points(frame, lm)
            if remaining > 0:
                writer.writerow([recording] + extract_features(lm))
                counts[recording] += 1
                remaining -= 1
            status = f"Recording {recording}: {remaining} left" if remaining else f"Ready - press one of: {ALLOWED}"
        else:
            status = "No hand visible"

        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        summary = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        cv2.putText(frame, summary, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Collect data (ESC to quit)", frame)

cap.release()
cv2.destroyAllWindows()
print("Saved to data.csv:", dict(counts))
