"""Live recognition using the trained model. Press ESC to quit."""
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np

from classifier import knn_predict
from features import create_detector, draw_points, extract_features, open_camera

CONFIDENCE = 0.6       # below this, show "..."
SMOOTHING_FRAMES = 8

data = np.load("model.npz")
X_train, y_train = data["X"], data["y"]
detector = create_detector()
cap = open_camera()
history = deque(maxlen=SMOOTHING_FRAMES)
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(img, int((time.time() - start) * 1000))

    shown, conf = "...", 0.0
    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        draw_points(frame, lm)
        x = np.array(extract_features(lm), dtype=np.float32)
        label, conf = knn_predict(X_train, y_train, x)
        history.append(label if conf >= CONFIDENCE else "...")
        shown = Counter(history).most_common(1)[0][0]
    else:
        history.clear()

    cv2.rectangle(frame, (20, 20), (280, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Letter: {shown}", (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"confidence: {conf:.2f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Sign Language - trained model", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
