"""
An *expert* classifier which uses hardcoded features to generate labels. Features
are determined at design time by humans. There is no learning/training phase.
"""
from typing import Iterable
import numpy as np
from scipy import ndimage
from naiveclassifier import NaiveClassifier



class ExpertClassifier(NaiveClassifier):
    """
    Implements hand-coded classifier based on human experience.
    """

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)


    def train(self, X: Iterable[np.ndarray], Y: np.ndarray):
        pass


    def predict(self, x: np.ndarray) -> str:
        """
        Takes an array representing pixel intensities, measures some statistics,
        and uses hard-coded thresholds to determine label.

        Args:
        * x (np.ndarray): An array of image pixel intensities.

        Returns:
        * A string label corresponding to one of art styles.
        """
        avg = self.avg_color(x)
        blur = self.blurriness(x, 1.0)
        var = self.variance(x)

        if blur < 100 and var > 4500:
            return 'Romanticism'
        elif blur < 100 and avg < 80:
            return 'Baroque'
        else:
            return 'Impressionism'


    def avg_color(self, x: np.array, xmin: int=None, xmax: int=None, \
                  ymin: int=None, ymax: int=None) -> float:
        return np.mean(x[ymin:ymax, xmin:xmax])
    

    def blurriness(self, x: np.array, std: float, xmin: int=None, xmax: int=None, \
                   ymin: int=None, ymax: int=None) -> float:
        x = x[ymin:ymax, xmin:xmax]
        npix = np.prod(x.shape)
        blurred = ndimage.filters.gaussian_filter(x, std)
        return np.sum(np.abs(x - blurred)) / npix


    def variance(self, x: np.array, xmin: int=None, xmax: int=None, \
                 ymin: int=None, ymax: int=None) -> float:
        return ndimage.measurements.variance(x[ymin:ymax, xmin:xmax])
