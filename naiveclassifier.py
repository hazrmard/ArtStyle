"""
The basic classifier class which learns the most frequent label in a dataset.
"""
from typing import Iterable, Dict, Any
import numpy as np


class NaiveClassifier:
    """
    Implements majority label classification as a baseline performance metric
    for later classifiers.

    Args:
    * Any keyword arguments containing hyperparameters.

    Attributtes:
    * hyperparameters (Dict[str, Any]): A dictionary of hyperparameters.
    """

    def __init__(self, **hyperparameters):
        self.hyperparameters = hyperparameters

    def train(self, X: Iterable[np.ndarray], Y: Iterable[int], **hyperparameters):
        """
        Train the classifier from features (X) and labels (Y).

        Args:
        * X (Iterable[np.ndarray]): A sequence of instances represented as arrays.
        * Y (Iterable[int]): A sequence of labels for corresponding instances.
        """
        self.hyperparameters.update(hyperparameters)
        (_, idx, counts) = np.unique(Y, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode = Y[index]
        self.majority_label = mode
    

    def predict(self, x: np.ndarray) -> int:
        """
        Predicts label of a single instance.

        Args:
        * x (np.ndarray): A single instance represented as an array.

        Returns:
        * The predicted label (int).
        """
        return self.majority_label
    

    def evaluate(self, X: Iterable[np.ndarray], Y: Iterable[int]) -> float:
        """
        Given instances and labels, returns the percentage of correct predictions.

        Args:
        * X (Iterable[np.ndarray]): A sequence of instances represented as arrays.
        * Y (Iterable[int]): A sequence of labels for corresponding instances.

        Returns:
        * The accuracy of prediction (float).
        """
        correct = sum([1 if self.predict(x)==y else 0 for x, y in zip(X, Y)])
        return correct / len(X)