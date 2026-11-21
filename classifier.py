from collections import Counter

import numpy as np


def knn_predict(X_train, y_train, x, k=7):
    """k-nearest-neighbours using only NumPy. Returns (label, confidence)."""
    dist = np.linalg.norm(X_train - x, axis=1)
    nearest = np.argsort(dist)[:k]
    label, votes = Counter(y_train[nearest]).most_common(1)[0]
    return label, votes / k
