"""
Defines the network, dataloader, and model for the cropped data set using
WikiArt paintings.
"""
import os
import csv
from typing import Callable

import torch
from torch.utils import model_zoo
from torchvision.models.alexnet import model_urls
from torchvision.models.alexnet import AlexNet as OldAlexNet

from .mnist import Model
from .loaders import ImageDataSet, ImageStreamer


# The column headers in the csv file describing images.
FNAME_COL = 'filename'
LABEL_COL = 'style'



class AlexNet(OldAlexNet):
    """
    Modified version of AlexNet with 136 classes to work with the WikiArt dataset.
    If `pretrained=True` loads all AlexNet weights except the very last linear
    layer which has 136 units.

    During training, does not modify convolutional layer weights.

    Args:

    * model_dir (str): Save path for pre-trained weights for AlexNet.
    * pretrained (bool): Whether to download and use pre-trained weights
    """

    def __init__(self, model_dir: str = None, pretrained: bool = False):
        super().__init__(num_classes=136)
        # load weights for old
        if pretrained:
            state = model_zoo.load_url(model_urls['alexnet'],
                                        model_dir=model_dir, map_location='cpu')
            # replace OldAlexNet final layer parameters with those for 136 classes
            state['classifier.6.weight'] = self.state_dict()['classifier.6.weight']
            state['classifier.6.bias'] = self.state_dict()['classifier.6.bias']
            self.load_state_dict(state)


    def forward(self, x):
        # do not backpropagate gradients to feature detectors
        with torch.no_grad():
            x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x



class Data(ImageDataSet):
    """
    A `DataSet` instance that iterates over train/test images. Requies the
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
    `PIL.Image` to `torch.Tensor`.
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

        super().__init__(images=image_stream, labels=labels, encode=encode,
            binarize=binarize, num_output_channels=3, transform=transform)
        