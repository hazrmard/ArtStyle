"""
Defines some IO functions to help with reading large number of files.
"""
import os
from typing import Iterable, Set, List, Tuple
from configparser import ConfigParser, ExtendedInterpolation
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np
from imageio import imread
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from sklearn.preprocessing import LabelBinarizer



class ImageStreamer(list):
    """
    Lazily loads images as numpy arrays. Supports list indexing operations.

    * ImageStreamer[integer] -> np.ndarray
    * ImageStreamer[slice] -> ImageStreamer

    Args:

    * `fnames (Iterable[str])`: A sequence of filenames to load.
    * `root (str)`: The root directory containing the filenames.
    """

    def __init__(self, fnames: Iterable[str], root: str=''):
        self.root = root
        super().__init__(fnames)


    def __getitem__(self, x):
        fnames = super().__getitem__(x)
        if isinstance(fnames, list):
            return ImageStreamer(fnames, root=self.root)
        elif isinstance(fnames, str):
            path = os.path.join(self.root, fnames)
            return imread(path)
    

    def __iter__(self):
        for fname in super().__iter__():
            yield imread(os.path.join(self.root, fname))



class ImageDataSet(Dataset):
    """
    A PyTorch compatible dataset. Lazily reads images and their labels. Supports
    list indexing. Returns a tuple of:

    * `torch.Tensor`: the image array,
    * `str`: the image label

    Args:

    * `images (ImageStreamer)`: The set of images to load as an `ImageStreamer`.
    * `labels (List[str])`: A sequence of labels for each image. The string
    labels are one-hot coded into an array. If a numpy array is passed, the labels
    are used as-is.

    Attributes:

    * `label_encoder (sklearn.preprocessing.LabelBinarizer)`: Encodes categorical
    labels in a one-hot format. E.g. ['a', 'b'] => [[1,0],[0,1]]
    """

    def __init__(self, images: ImageStreamer, labels: Iterable[str]):

        self.label_encoder = None
        if isinstance(labels, np.ndarray):
            self.labels = Tensor(labels)
        else:
            self.label_encoder = LabelBinarizer()
            self.labels = Tensor(self.label_encoder.fit_transform(labels))

        self.images = images

        if len(labels) != len(images):
            raise ValueError('Number of images ({:d}) != number of labels({:d})'
                             .format(len(images), len(labels)))


    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        return (Tensor(self.images[idx]), self.labels[idx])


    def __len__(self) -> int:
        return len(self.labels)



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
        avail_fnames = fnames & subset
    else:
        avail_fnames = fnames
    return ImageStreamer(avail_fnames, root=basedir)



def get_config(*fnames: str) -> namedtuple:
    """
    Reads the config file (.ini) and returns a `namedtuple` where config
    properties can be accessed by name: `config.prop1...`,
    """
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(fnames)

    sections = parser.sections()
    Config = namedtuple('Config', sections)
    ConfigDict = {}
    for s in sections:
        options = {}
        for opt, val in parser.items(s):
            if s == 'paths':
                val = os.path.expanduser(val)
                val = os.path.expandvars(val)
            else:
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            options[opt] = val
        ntpl = namedtuple(s, options.keys())
        ConfigDict[s] = ntpl(**options)
    return Config(**ConfigDict)
