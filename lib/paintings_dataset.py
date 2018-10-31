import numpy as np
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd


SCRIPT_FNAME = os.path.basename(__file__)

# class PaintingsDataset(Dataset):
#     """Aperture domain dataset."""
#
#     def __init__(self, fname, num_samples):
#         """
#         Args:
#             fname: file name for aperture domain data
#             num_samples: number of samples to use from data set
#         """
#
#         print('{}: fname = {}, num_samples = {}'.format(SCRIPT_FNAME, fname, num_samples))
#         self.fname = fname
#
#         # check if files exist
#         if not os.path.isfile(fname):
#             raise IOError('{}: {} does not exist.'.format(SCRIPT_FNAME, fname))
#
#
#         # convert data to single precision pytorch tensors
#         self.data_tensor = torch.from_numpy(inputs).float()
#         self.target_tensor = torch.from_numpy(targets).float()
#
#     def __len__(self):
#         return self.data_tensor.size(0)
#
#     def __getitem__(self, idx):
#         return self.data_tensor[idx], self.target_tensor[idx]


class PaintingsDataset(Dataset):
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
    * `num_output_channels (int)`: 1 or 3. Number of channels in output tensor for
    grayscale input images. If 3, then grayscale images are converted to 3 channels
    to be consistent with RGB images.

    Attributes:

    * `encoder (sklearn.preprocessing.LabelEncoder)`: Encodes string labels into
    integers. E.g ['a', 'b'] => [0, 1]
    * `binarizer (sklearn.preprocessing.LabelBinarizer)`: Encodes categorical
    labels in a one-hot format. E.g. [0, 1] => [[1,0],[0,1]]
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
        # 1 channel grayscale images are converted to 3 channel grayscale images,
        # or 3 channel RGBs to 1 channel grayscale so RGB/Grayscale images can
        # be read together as Channels x Height x Width tensors.
        img = self.images[idx]
        if (img.mode in ('1', 'L', 'P') and self.num_output_channels == 3) or\
            (img.mode == 'RGB' and self.num_output_channels == 1):
            img = self.grayscale(img)
        if (img.mode in ('RGBA','CMYK')):
            img = img.convert('RGB')
        img_tensor = self.transformer(img)

        return (img_tensor,
                torch.tensor(self.labels[idx], dtype=torch.long))


    def __len__(self) -> int:
        return len(self.labels)
