"""Build model.npz from data.csv (pure NumPy, no scikit-learn needed)."""
import csv
from collections import Counter

import numpy as np

from classifier import knn_predict

labels, rows = [], []
with open("data.csv") as f:
    for r in csv.reader(f):
        labels.append(r[0])
        rows.append([float(v) for v in r[1:]])

y_all = np.array(labels)
X_all = np.array(rows, dtype=np.float32)
counts = Counter(y_all)
print("Samples per letter:", dict(sorted(counts.items())))

# Ignore letters with too few samples
keep = np.array([counts[l] >= 30 for l in y_all])
X, y = X_all[keep], y_all[keep]
if len(set(y)) < 2:
    raise SystemExit("Need at least 2 letters with 30+ samples each.")
print("Letters used:", sorted(set(y)))

# Quick accuracy check on a random 20% of the data
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split = int(len(X) * 0.8)
train_i, test_i = idx[:split], idx[split:]
correct = sum(knn_predict(X[train_i], y[train_i], X[i])[0] == y[i] for i in test_i)
print(f"Test accuracy: {correct / len(test_i) * 100:.1f}%")

np.savez("model.npz", X=X, y=y)
print("Saved model.npz")
