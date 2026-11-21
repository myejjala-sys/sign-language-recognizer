# How It Works: Build This Project Yourself

This guide explains the ideas behind the sign language recognizer and walks you through
building it from an empty folder. You do not need machine learning experience. You need
basic Python and a webcam.

## Contents

1. [The big picture](#1-the-big-picture)
2. [Concept: landmarks](#2-concept-landmarks)
3. [Concept: why simple rules fail](#3-concept-why-simple-rules-fail)
4. [Concept: normalization](#4-concept-normalization)
5. [Concept: k-nearest neighbours](#5-concept-k-nearest-neighbours)
6. [Concept: confidence and smoothing](#6-concept-confidence-and-smoothing)
7. [Build it step by step](#7-build-it-step-by-step)
8. [Understanding accuracy](#8-understanding-accuracy)
9. [Troubleshooting](#9-troubleshooting)
10. [Challenges to go further](#10-challenges-to-go-further)

---

## 1. The big picture

The program has two phases:

```
TRAINING PHASE (you do this once)          LIVE PHASE (every time you run it)

 webcam -> landmarks -> features            webcam -> landmarks -> features
              |                                          |
              v                                          v
      save with a letter label              compare with saved examples
              |                                          |
              v                                          v
          data.csv  --->  model.npz  --->        predicted letter
```

- **Training phase:** you show a sign, tell the program which letter it is, and it
  stores the numbers describing your hand. After many examples, it knows what each
  letter looks like.
- **Live phase:** for each camera frame, it describes your hand with numbers and finds
  the most similar saved examples. Their letter is the prediction.

This idea (learn from labelled examples instead of writing rules) is called
**supervised machine learning**.

## 2. Concept: landmarks

A camera frame is just a grid of pixels, which is hard to reason about. **MediaPipe**
is a library that finds a hand and returns **21 landmarks**: points on the wrist and
on each finger joint. Each point has `x` and `y` between 0 and 1 (position in the
image).

```
Landmark numbers (the important ones)

 0  = wrist
 1-4   = thumb   (4 = thumb tip)
 5-8   = index   (5 = knuckle, 8 = tip)
 9-12  = middle  (9 = knuckle, 12 = tip)
13-16  = ring    (13 = knuckle, 16 = tip)
17-20  = pinky   (17 = knuckle, 20 = tip)
```

Now a hand sign is just **21 points = 42 numbers**. Much easier to work with than
pixels. You can print them and see them change as you move your fingers.

**Try it:** run a short script that prints `landmarks[8].x` and `landmarks[8].y` (index
fingertip). Move your finger and watch the numbers change.

## 3. Concept: why simple rules fail

The first version of this project used rules like:

```python
index_up = landmarks[8].y < landmarks[6].y   # fingertip above the middle joint
```

and then `if index_up and not middle_up ...: letter = "D"`.

This works for a few signs but fails because:

- Many letters have the **same finger pattern** (A, E, M, N, S, T are all fists).
- Fixed thresholds depend on **distance from the camera** and **hand size**.
- Rules **fight each other**: adding one rule breaks another.
- J and Z need **motion**, so no static rule can recognize them.

Learning from examples avoids writing and maintaining these rules.

## 4. Concept: normalization

If you move your hand left, or hold it closer, all 42 numbers change, even though the
sign is the same. To fix this, we describe the hand **relative to itself**:

1. **Subtract the wrist position** from every point. Now the wrist is at (0, 0), so
   where the hand is in the picture no longer matters.
2. **Divide by the largest distance from the wrist.** Now a big hand (close to the
   camera) and a small hand (far away) give similar numbers.

```python
import math

def extract_features(lm):
    wrist = lm[0]
    coords = [(p.x - wrist.x, p.y - wrist.y) for p in lm]
    scale = max(math.hypot(x, y) for x, y in coords) or 1e-6
    feats = []
    for x, y in coords:
        feats += [x / scale, y / scale]
    return feats          # 42 numbers
```

Good features matter more than a fancy model. This step is why the project works at
different distances.

## 5. Concept: k-nearest neighbours

**k-NN** is one of the simplest ML ideas: *to classify something new, find the k most
similar examples you have already seen, and let them vote.*

Imagine every saved sample as a dot in 42-dimensional space. Samples of the same letter
form a cluster. A new hand is another dot; the letter whose cluster it lands in wins.

```python
import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, x, k=7):
    dist = np.linalg.norm(X_train - x, axis=1)   # distance to every saved sample
    nearest = np.argsort(dist)[:k]               # indexes of the k closest
    label, votes = Counter(y_train[nearest]).most_common(1)[0]
    return label, votes / k                      # letter and confidence
```

How to read it:

- `dist` measures how different the new hand is from each saved sample.
- `nearest` picks the k smallest distances.
- The most common letter among them is the prediction.
- `votes / k` is the **confidence**: 7 out of 7 neighbours agreeing gives 1.0.

**Why k-NN here?**

- No training step: "training" just means saving your examples.
- It works well with a few hundred samples per letter.
- It only needs NumPy, so it runs on computers that block other libraries.

**Weaknesses:** it gets slower with a lot of data, and it only knows what you recorded.
(Later you can try a neural network.)

## 6. Concept: confidence and smoothing

Two small tricks make the app feel stable:

**Confidence threshold.** If fewer than 60% of neighbours agree, show `...` instead of
guessing. A wrong letter is worse than no letter.

**Smoothing.** Predictions flicker frame to frame. Keep the last 8 and show the most
common:

```python
from collections import deque, Counter
history = deque(maxlen=8)
history.append(label)
shown = Counter(history).most_common(1)[0][0]
```

A bigger window is steadier but reacts more slowly.

## 7. Build it step by step

### Step 0: Set up

```
mkdir sign-language-recognizer
cd sign-language-recognizer
python -m venv venv
venv\Scripts\activate
python -m pip install opencv-python mediapipe
```

Use `python -m pip` (not just `pip`); it avoids problems on locked-down computers.

### Step 1: Show the camera (check your camera works)

```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)          # mirror view
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:     # ESC
        break
cap.release()
cv2.destroyAllWindows()
```

Concepts: a video is a loop of frames. `waitKey` also reads the keyboard.

### Step 2: Detect the hand

Add MediaPipe's hand landmarker: download `hand_landmarker.task`, create a
`HandLandmarker` in **VIDEO** mode (it tracks the hand between frames, which is faster
than detecting from scratch every time), and draw the 21 points on the frame. In this
repo, see `create_detector()` and `draw_points()` in `features.py`.

Goal: you see green dots on your hand.

### Step 3: Turn the hand into numbers

Write `extract_features(landmarks)` (section 4). Print the 42 numbers and check that
they stay similar when you move your hand closer or farther away with the same sign.

### Step 4: Collect labelled data (`collect.py`)

For each frame with a visible hand:

1. If you pressed an allowed letter key, start recording for that letter.
2. Append one CSV row: `letter, f0, f1, ... f41`.
3. Stop after 100 rows and wait for the next key.

Result: `data.csv`, one row per sample. Open it in a text editor to see what the data
looks like.

**Data tips (most important part of the project):**

- 200+ samples per letter.
- Move the hand a little: closer, farther, tilted, slightly rotated.
- Keep the sign correct. One wrong sample is worse than missing samples.
- Keep similar letters balanced (same number of samples).

### Step 5: Build the model (`train.py`)

1. Read `data.csv` into `X` (features) and `y` (letters).
2. Hold out 20% as a test set, and measure how often k-NN predicts those correctly.
3. Save `X` and `y` with `np.savez("model.npz", X=X, y=y)`.

### Step 6: Predict live (`predict.py`)

Same camera loop as before, plus:

1. Load `model.npz`.
2. Each frame: `extract_features` -> `knn_predict` -> apply threshold -> smooth.
3. Draw the letter and confidence on the screen.

### Step 7: Improve it

Add letters 3 to 5 at a time, test, and re-collect for the confusing ones. Commit to
Git after each working step.

## 8. Understanding accuracy

`train.py` prints an accuracy such as 98%. Be careful:

- Neighbouring video frames look almost identical, so test samples often have a twin in
  the training set. This makes the number **optimistic**.
- The real test is **live use**: does it work when you hold the sign again an hour later?
- If a letter is often confused with another, record more samples of **both** at varied
  angles. If they are physically the same shape from the front, a single camera angle
  can't separate them.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Letters flicker | No smoothing | Increase `SMOOTHING_FRAMES` |
| Shows `...` all the time | Confidence too strict, or too little data | Lower `CONFIDENCE`, record more samples |
| Wrong letter often | Messy training data | `del data.csv`, re-record carefully |
| Works only close to the camera | Data recorded at one distance | Record at several distances |
| Works for you, not a friend | Model learned your hand | Record samples from them too |
| `No module named ...` | venv not active | `venv\Scripts\activate` |
| `Application Control policy` error | Computer blocks compiled DLLs | Use pure-Python/NumPy code, a personal PC, or ask IT |
| Slow video | High resolution | Use 640x480 and VIDEO mode |

## 10. Challenges to go further

1. **Word builder:** if the same letter is held for 1 second, append it to a sentence.
2. **Confusion matrix:** print which letters are mistaken for which in `train.py`.
3. **Both hands:** set `num_hands=2` and handle left and right separately.
4. **Neural network:** replace k-NN with a small network (needs a library such as
   scikit-learn or PyTorch, if your computer allows it) and compare accuracy and speed.
5. **Motion letters:** record sequences of 20 frames to recognize J and Z.
6. **Web or mobile version:** MediaPipe also runs in the browser with JavaScript.
7. **Write tests:** `knn_predict` is a pure function, so it is easy to unit test.

## Key ideas to remember

- Good **features** (normalized landmarks) matter more than a complicated model.
- **Data quality** decides accuracy: clean, varied, balanced samples.
- Start **small** (3 letters), make it work, then grow.
- Always check **live**, not only the accuracy number.
