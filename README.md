# Sign Language Alphabet Recognizer

Real-time hand sign recognition from a webcam, using **OpenCV**, **MediaPipe** and a
classifier trained on **your own hand data**.

> Want to understand how it works or build it yourself from scratch?
> Read **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)**.

## How it works (short version)

1. MediaPipe finds **21 landmarks** (points) on your hand in every camera frame.
2. The points are **normalized** so position and hand size don't matter.
3. A **k-nearest-neighbours** classifier (pure NumPy) compares them with your
   recorded examples and predicts the letter.
4. The last few predictions are **smoothed** so the letter doesn't flicker.

## Project files

| File | Purpose |
|---|---|
| `features.py` | Camera setup, MediaPipe detector, landmark normalization |
| `collect.py` | Record training samples from your webcam into `data.csv` |
| `train.py` | Check accuracy and build `model.npz` from `data.csv` |
| `classifier.py` | The k-nearest-neighbours classifier (NumPy only) |
| `predict.py` | Live recognition using the trained model |
| `requirements.txt` | Python libraries needed |

## Setup

```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

(On macOS/Linux, activate with `source venv/bin/activate`.)

The hand model (`hand_landmarker.task`, about 8 MB) downloads automatically the first
time you run a script, so you need an internet connection once.

## Usage

Run these **one at a time**, in this order:

```
python collect.py     # 1. record samples
python train.py       # 2. build the model
python predict.py     # 3. live recognition
```

### 1. Collect data

- Choose your letters at the top of `collect.py`:
  ```python
  ALLOWED = "ABL"
  ```
- Show a sign, click the camera window, then press that letter key **once**.
  It records 100 samples automatically. Move your hand slightly while it records.
- Press the same key again for more samples (200 or more per letter is better).
- Press **ESC** to quit. Data is added to `data.csv`.

### 2. Train

```
python train.py
```

Prints the accuracy and saves `model.npz`. Letters with fewer than 30 samples are
ignored.

### 3. Predict

```
python predict.py
```

Press **ESC** to quit. Below the confidence threshold the screen shows `...`.

## Reset and start over

If your data got messy (wrong keys pressed, wrong sign shown), delete it and record again:

```
del data.csv
del model.npz
```

(On macOS/Linux use `rm data.csv model.npz`.)

Then run `collect.py` again. Deleting `data.csv` only removes your recorded samples.
The code and the downloaded hand model are not affected.

To forget just one letter, open `data.csv` in a text editor and delete the lines that
start with that letter, or reset everything and record again.

## Tips for good accuracy

- Use the **same hand**, the same distance (40 to 60 cm) and the same lighting
  every time. Light in front of you, not behind you.
- Record each letter at slightly different angles and distances.
- Add **3 to 5 letters at a time** and test before adding more.
- Similar letters (for example M, N, S, T) need more samples each.
- Use correct ASL hand shapes. Check an ASL alphabet chart.

## Limitations

- **J and Z** need motion, so a single-frame classifier can't learn them.
- The model learns from *your* hand, so it works best for you.
- One hand at a time.

## Common problems

| Problem | Fix |
|---|---|
| `No module named cv2` | Activate the venv: `venv\Scripts\activate`, then install requirements |
| `pip.exe` blocked | Use `python -m pip install ...` instead of `pip install ...` |
| `Application Control policy has blocked this file` | Your PC blocks some compiled libraries. This project avoids scikit-learn for that reason. Try a personal computer, or ask the IT admin |
| Keys do nothing in `collect.py` | Click the camera window first, and make sure the letter is in `ALLOWED` |
| `train.py` says too few samples | Record at least 30 (ideally 200+) per letter |
| `predict.py` can't find `model.npz` | Run `train.py` first |
| Camera is black or in use | Close Zoom, Teams or browser tabs that use the webcam |

## Ideas to extend it

- Build words: add a letter to a sentence when a sign is held for 1 second.
- Record more letters and compare accuracy.
- Replace k-NN with a small neural network.
- Add a screenshot or GIF of the app to this README.
