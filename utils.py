"""
Defines some IO functions to help with reading large number of files.
"""
from typing import Iterable
import matplotlib.pyplot as plt
from imageio import imread
import numpy as np

class ImageStreamer(list):
    """
    Overloads a python list to iterate over image files as numpy arrays. The
    class is instantiated with a list of image file paths. When an image is
    indexed by an integer, a numpy array of the corresponding np.ndarray is
    returned.

    * ImageStreamer[integer] -> np.ndarray
    * ImageStreamer[slice] -> ImageStreamer
    """

    def __init__(self, fnames: Iterable[str]):
        super().__init__(fnames)
    

    def __getitem__(self, x):
        fnames = super().__getitem__(x)
        if isinstance(fnames, list):
            return ImageStreamer(fnames)
        elif isinstance(fnames, str):
            return imread(fnames)
        else:
            print('Hey', x)
    

    def __iter__(self):
        for fname in super().__iter__():
            yield imread(fname)




def plot_cmatrix(cmatrix: np.ndarray, labels: Iterable[str]=[], cbar: bool=True, norm=False):
    if norm:
        im = plt.imshow(cmatrix, cmap=plt.cm.get_cmap('Blues'), vmin=0, vmax=1)
    else:
        im = plt.imshow(cmatrix, cmap=plt.cm.get_cmap('Blues'))  
    if cbar:
        plt.gcf().colorbar(im)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    return im
