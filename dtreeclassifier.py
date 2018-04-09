"""
A decision tree classifier finds best features to split along until a split
contains an acceptable number of labels of the same class.
"""
from typing import Iterable
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from naiveclassifier import NaiveClassifier



class DTreeClassifier(DecisionTreeClassifier, NaiveClassifier):
    """
    Implements a decision tree classifier based on sklearn.tree.DecisionTreeClassifier.
    """

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)


    def train(self, X: np.ndarray, Y: np.ndarray):
        """
        Training involves finding useful splits across features.

        Args:
        * X (np.ndarray): A 2D array where each row is the features corresponding
        to each label in Y.
        * Y (List): A list of corresponding string labels for features.
        """
        self.fit(X, Y)


    def predict(self, x: np.ndarray) -> str:
        """
        Takes an array representing various image features and predicts a label.

        Args:
        * x (np.ndarray): An array of features for a single image sample.

        Returns:
        * A string label corresponding to one of art styles.
        """
        return super().predict(np.reshape(x, (1, -1)))[0]
