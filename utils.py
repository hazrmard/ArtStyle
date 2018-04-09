"""
Defines some IO functions to help with reading large number of files.
"""
from typing import Iterable, Callable, Set
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
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

    def __init__(self, fnames: Iterable[str], basedir: str=''):
        self.basedir = basedir
        super().__init__(fnames)
    

    def __getitem__(self, x):
        fnames = super().__getitem__(x)
        if isinstance(fnames, list):
            return ImageStreamer(fnames, basedir=self.basedir)
        elif isinstance(fnames, str):
            path = os.path.join(self.basedir, fnames)
            return imread(path)
        else:
            print('Hey', x)
    

    def __iter__(self):
        for fname in super().__iter__():
            yield imread(os.path.join(self.basedir, fname))



def plot_cmatrix(cmatrix: np.ndarray, labels: Iterable[str]=(), cbar: bool=True, \
    norm=False) -> AxesImage:
    """
    Draws a heatmap of a confusion matrix.

    Args:
    * cmatrix (np.ndarray): The confusion matrix.
    * labels (Iterable[str]): A sequence of labels to put on each axis corresponding
    to the classes. If empty, no labels placed.
    * cbar (bool): Whether to draw a colorbar with the heatmap.
    * norm (bool): Whether to normalize all values between 0 and 1.

    Returns:
    The axis on which the heatmap was drawn.
    """
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



def get_image_stream(basedir: str, subset: Set[str]=None) -> ImageStreamer:
    """
    Creates an ImageStreamer from all images in a directory optionally filtered
    using a subset.

    Args:
    * basedir (str): Directory containing images.
    * subset (Set): Set of images to include - may or may not exist in basedir. If
    empty, all images in basedir are used.

    Returns:
    * An ImageStreamer from the images in basedir optionally filtered.
    """
    fnames = set(os.listdir(basedir))
    if subset is not None:
        avail_fnames = fnames & subset          # downloaded filenames in training split
    else:
        avail_fnames = fnames
    return ImageStreamer(avail_fnames, basedir=basedir)