import os
from typing import Iterable, Set, List, Tuple

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