# Sign Language Alphabet Recognizer

Real-time hand sign recognition using OpenCV and MediaPipe.
# Sign Language Alphabet Recognizer

Real-time hand sign recognition using OpenCV and MediaPipe,
with a classifier trained on my own hand data.

## How it works
1. MediaPipe detects 21 hand landmarks.
2. Landmarks are normalized (position and hand size).
3. A k-nearest-neighbours classifier (NumPy) predicts the letter.

## Setup
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt

## Usage
python collect.py   # record samples (press a letter key once)
python train.py     # build model.npz
python predict.py   # live recognition (ESC to quit)

## Limitations
J and Z need motion, so they are not supported.