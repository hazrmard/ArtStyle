"""
Loades cropped images from training/evaluation.
"""
from .cropped import CroppedData


class ResizedData(CroppedData):

    subdir = 'resized'