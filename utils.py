"""
Defines some IO functions to help with reading large number of files.
"""
import os
from typing import Iterable, Set
from configparser import ConfigParser, ExtendedInterpolation
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import numpy as np

from lib.loaders import ImageStreamer



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
