"""
An *expert* classifier which uses hardcoded features to generate labels. Features
are determined at design time by humans. There is no learning phase.
"""
from typing import Iterable, Tuple
import numpy as np
from scipy import ndimage
from naiveclassifier import NaiveClassifier



class ExpertClassifier(NaiveClassifier):
    """
    Implements hand-coded classifier based on human experience.
    """

    def __init__(self, **hyperparameters):
        super().__init__(**hyperparameters)
    

    @property
    def xdim(self) -> int:
        return self.hyperparameters.get('shape')[0]
    

    @property
    def ydim(self) -> int:
        return self.hyperparameters.get('shape')[1]


    def train(self, X: Iterable[np.ndarray], Y: np.ndarray, **hyperparameters):
        pass


    def predict(self, x: np.ndarray) -> int:
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
