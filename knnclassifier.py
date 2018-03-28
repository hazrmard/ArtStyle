"""
A supervised classifier which uses k-Nearest Neighbours to vote on a prediction
of style for a painting. The nearest beighbours are calculated in feature-space.
As input, the classifier takes pre-computed feautres from images (and not raw
pixel arrays).
"""
from typing import Iterable, List
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from naiveclassifier import NaiveClassifier



class KNNClassifier(KNeighborsClassifier, NaiveClassifier):
    """
    Implements a k-Nearest Neighbours classifier. Inherits from
    scipy.neighbors.KNeighborsClassifier.
    """

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)


    def train(self, X: np.ndarray, Y: List):
        """
        Training involves simply memorizing features for training data.

        Args:
        * X (np.ndarray): A 2D array where each row is the features corresponding
        to each label in Y.
        * Y (List): A list of corresponding string labels for features.
        """
        self.fit(X, Y)
    

    def predict(self, x: np.ndarray) -> str:
        return super().predict(np.reshape(x, (1, -1)))[0]
