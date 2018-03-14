from typing import Iterable
import numpy as np
from naiveclassifier import NaiveClassifier



class ExpertClassifier(NaiveClassifier):

    def train(self, X: Iterable[np.ndarray], Y: np.ndarray, **hyperparameters):
        pass
    

    def predict(self, X):
        pass