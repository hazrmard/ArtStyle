import os
import csv
from typing import Callable


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from sklearn.preprocessing import LabelBinarizer
import numpy as np


from torchvision import transforms

from .loaders import ImageDataSet, ImageStreamer


# The column headers in the csv file describing images.
FNAME_COL = 'filename'
LABEL_COL = 'style'


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
        
class Model():
    """
    The `Model` encapsulates the network, the dataset, and the training/evaluation
    loop.

    Args:

    * `net (torch.nn.Module)`: The network to train.
    * `criterion (torch.nn.modules.loss._Loss)`: The loss function.
    * `optimizer (torch.optim.Optimizer)`: The learning algorithm instantiated with
    `net.parameters()`.
    * `cuda (bool)`: Whether to use a GPU (if available).
    """

    def __init__(self, net: nn.Module, criterion: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer, cuda: bool):
        
        self.net = net
        # pylint: disable=E1101
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and cuda)\
                                    else "cpu")
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, dataset: torch.utils.data.Dataset, batch_size: int=10,
        epochs: int=1):
        """
        Train the network using provided hyperparameters.

        Args:

        * `dataset (torch.utils.data.Dataset)`: The dataset object containing instances.
        * `batch_size (int)`: Number of instances per minibatch.
        * `epochs (int)`: Number of times to iterate over dataset.
        """

        trainloader = DataLoader(dataset, batch_size=batch_size)
        self.net.to(self.device)

        for epoch in range(epochs):
            print('Epoch #', epoch)
            running_loss = 0.
            for i, (batchX, batchY) in enumerate(trainloader, 1):
                batchX, batchY = batchX.to(self.device), batchY.to(self.device)
                
                self.optimizer.zero_grad()           # clear gradients from prev. step
                predY = self.net(batchX)             # get predicted labels
                loss = self.criterion(predY, batchY) # compute loss
                loss.backward()                      # backpropagate - populate error grad.
                self.optimizer.step()                # update weights based on gradients
                
                running_loss += loss.item()
                if i % 1000 == 0:
                    print('Min-batch # {0:4d}\t Loss: {1:3.3f}'.format(i, running_loss / 1000))
                    running_loss = 0.
        