"""
Loads cropped images from training/test sets.
"""
import os
import csv
from typing import Callable

from torchvision.transforms import Compose, Normalize, ToTensor

from .loaders import ImageDataSet, ImageStreamer


# The column headers in the csv file describing images.
FNAME_COL = 'filename'
LABEL_COL = 'style'



class CroppedData(ImageDataSet):
    """
    A `DataSet` instance that iterates over train/test images. Requires the
    root directory structure to be:

        root/
            train/
                cropped/
                resized/
            test/
                cropped/
                resized/

    The csv file containing labels should be of the form (column ordering does
    not matter):

        filename    |   style
        -------------------
        fname1.jpg  |   style1

    Args:

    * `root (str)`: Path to root directory.
    * `info_csv (str)`: Path to csv file containing filenames and labels.
    * `train (bool)`: Whether to load training or testing datasets.
    * `encode (bool)`: Whether to encode string labels to integers.
    * `binarize (bool)`: Whether to one-hot code integer labels to vectors.
    * `transform (Callable)`: A function that transforms a PIL Image to tensor
    while also applying any other transformations. By if None, simply converts
    `PIL.Image` to `torch.Tensor` after normalization for AlexNet.
    """

    subdir = 'cropped'

    def __init__(self, root: str, info_csv: str, train: bool=True, encode: bool=True,
        binarize: bool = False, transform: Callable = None):

        if train:
            root = os.path.join(root, 'train', self.__class__.subdir)
        else:
            root = os.path.join(root, 'test', self.__class__.subdir)
        
        info = {}
        with open(info_csv, 'r', encoding='utf-8') as f:
            csvfile = csv.DictReader(f)
            for row in csvfile:
                info[row[FNAME_COL]] = row[LABEL_COL]

        # Files actually present in the root directory
        avail_fnames = set(os.listdir(root))
        # The set of files that can be read: files in csv and files available
        self.fnames = list(set(info.keys()) & avail_fnames)
        labels = [info[fname] for fname in self.fnames]

        image_stream = ImageStreamer(fnames=self.fnames, root=root)

        # Normalize image according to training set for torchvision models.
        # See https://pytorch.org/docs/stable/torchvision/models.html
        if transform is None:
            transform = Compose([ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])

        super().__init__(images=image_stream, labels=labels, encode=encode,
            binarize=binarize, num_output_channels=3, transform=transform)
        