from typing import Iterable
import numpy as np


class NaiveClassifier:

    def train(self, X: Iterable[np.ndarray], Y: np.ndarray, **hyperparameters):
        (_, idx, counts) = np.unique(Y, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode = Y[index]
        self.majority_label = mode
    

    def predict(self, X):
        return self.majority_label
    

    def evaluate(self, X, Y):
        correct = sum([1 if self.predict(x)==y else 0 for x, y in zip(X, Y)])
        return correct / len(X)