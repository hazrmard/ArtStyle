"""
Defines loader classes for image data:

* `ImageStreamer`: Instantiated with list of image filenames and root directory.
Lazily reads images when indexed. Can be iterated over.

* `ImageDataSet`: `torch.utils.data.DataSet` subclass. Returns a tuple of tensors
of the image and label. Instantiated with an `ImageStreamer`, and list of corresponding
labels in order. Can be indexed.
"""

import os
from typing import Iterable, Tuple, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import ToTensor, Grayscale
from sklearn.preprocessing import LabelBinarizer, LabelEncoder



class ImageStreamer:
    """
    Lazily loads images as PIL images. Supports list indexing operations.

    * ImageStreamer[integer] -> PIL.Image
    * ImageStreamer[slice] -> ImageStreamer

    Args:

    * `fnames (Iterable[str])`: A sequence of filenames to load.
    * `root (str)`: The root directory containing the filenames.
    """

    def __init__(self, fnames: Iterable[str], root: str=''):
        self.root = root
        self.fnames = fnames


    def __getitem__(self, x):
        fnames = self.fnames.__getitem__(x)
        if isinstance(fnames, list):
            return ImageStreamer(fnames, root=self.root)
        elif isinstance(fnames, str):
            path = os.path.join(self.root, fnames)
            return Image.open(path)


    def __iter__(self):
        for fname in self.fnames:
            yield Image.open(os.path.join(self.root, fname))


    def __len__(self):
        return len(self.fnames)



class ImageDataSet(Dataset):
    """
    A PyTorch compatible dataset. Lazily reads images and their labels. Supports
    list indexing. Able to load 1- or 3- channel images. Returns a tuple of:

    * `torch.Tensor`: Channels x Height x Width image array,
    * `torch.Tensor`: the image label

    Args:

    * `images (ImageStreamer)`: The set of images to load as an `ImageStreamer`.
    * `labels (List[str], np.ndarray)`: A sequence of labels for each image.
    * `encode (bool)`: Whether to encode string labels to integers.
    * `binarize (bool)`: Whether to one-hot code integer labels.
    * `transform (Callable)`: A function that transforms a PIL Image to tensor
    while also applying any other transformations. By if None, simply converts
    `PIL.Image` to `torch.Tensor`.
    * `num_output_channels (int)`: 1 or 3. Number of channels in output tensor.

    Attributes:

    * `encoder (sklearn.preprocessing.LabelEncoder)`: Encodes string labels into
    integers.
    * `binarizer (sklearn.preprocessing.LabelBinarizer)`: Encodes categorical
    labels in a one-hot format. E.g. ['a', 'b'] => [[1,0],[0,1]]
    """

    def __init__(self, images: ImageStreamer, labels: Iterable, encode: bool,
        binarize: bool, transform: Callable = None, num_output_channels=3):

        # pylint: disable=E1102,E1101
        self.transformer = ToTensor() if transform is None else transform
        self.encoder = LabelEncoder()
        self.binarizer = LabelBinarizer()
        self.num_output_channels = num_output_channels
        self.grayscale = Grayscale(num_output_channels=self.num_output_channels)

        if encode:
            labels = self.encoder.fit_transform(labels)
        if binarize:
            self.binarizer.fit(labels)
            self.labels = torch.tensor(self.binarizer.transform(labels), dtype=torch.long)
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)

        self.images = images

        if len(labels) != len(images):
            raise ValueError('Number of images ({:d}) != number of labels({:d})'
                             .format(len(images), len(labels)))


    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        # pylint: disable=E1102,E1101
        # Ensure that all images read have 3 channels. 1 channel grayscale images
        # are converted to 3 channel grayscale images.
        img = self.images[idx]
        if img.mode in ('1', 'L', 'P'):     # i.e. single channel
            img = self.grayscale(img)
        img_tensor = self.transformer(img)

        return (img_tensor,
                torch.tensor(self.labels[idx], dtype=torch.long))


    def __len__(self) -> int:
        return len(self.labels)