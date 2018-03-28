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

    def train(self, X: Iterable[np.ndarray], Y: Iterable[int]):
        """
        Train the classifier from features (X) and labels (Y).

        Args:
        * X (Iterable[np.ndarray]): A sequence of instances represented as arrays.
        * Y (Iterable[int]): A sequence of labels for corresponding instances.
        """
        (_, idx, counts) = np.unique(Y, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode = Y[index]
        self.majority_label = mode
    

    def predict(self, x: np.ndarray) -> str:
        """
        Predicts label of a single instance.

        Args:
        * x (np.ndarray): A single image instance represented as an array.

        Returns:
        * The predicted label i.e. the majority label in training data.
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
        predictions = [self.predict(x) for x in X]
        correct = sum([1 if pred==y else 0 for pred, y in zip(predictions, Y)])
        return correct / len(X), predictions