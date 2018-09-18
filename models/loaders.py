import os
from typing import Iterable, Set, List, Tuple

from imageio import imread
import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.transforms import ToTensor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder



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
    * `labels (List[str], np.ndarray)`: A sequence of labels for each image.
    * `encode (bool)`: Whether to encode string labels to integers.
    * `binarize (bool)`: Whether to one-hot code integer labels.

    Attributes:

    * `encoder (sklearn.preprocessing.LabelEncoder)`: Encodes string labels into
    integers.
    * `binarizer (sklearn.preprocessing.LabelBinarizer)`: Encodes categorical
    labels in a one-hot format. E.g. ['a', 'b'] => [[1,0],[0,1]]
    """

    def __init__(self, images: ImageStreamer, labels: Iterable, encode: bool,
        binarize: bool):

        # pylint: disable=E1102,E1101
        self.converter = ToTensor()
        self.encoder = LabelEncoder()
        self.binarizer = LabelBinarizer()
        
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
        return (self.converter(self.images[idx]), torch.tensor(self.labels[idx], dtype=torch.long))


    def __len__(self) -> int:
        return len(self.labels)