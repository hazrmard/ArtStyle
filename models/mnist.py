"""
Defines a prototypical model module. It contains:

* `Net (torch.nn.Module)`: A pytorch neural network definition.
* `Data (torch.utils.data.DataSet)`: A pytorch dataset to load instances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # in: N x 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 3, 5)
        # + relu
        # in: N x 3 x 24 x 24
        # + pooling (2, 2)
        # in: N x 3 x 12 x 12
        self.conv2 = nn.Conv2d(3, 3, 5)
        # + relu
        # in: N x 3 x 8 x 8
        # + pooling (2, 2)
        # in: N x 3 x 4 x 4
        # + view N x 48
        self.fc1 = nn.Linear(48, 10)
        # in: N x 10
        # + softmax


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = x.view(-1, 48)
        x = self.fc1(x)
        # x = F.softmax(x, dim=1)
        return x



class Data(MNIST):
    """
    The `torchvision.datasets.MNIST` class but with transforms predefined so
    the images are loaded as `1 x 28 x 28` tensors and labels as one-hot coded
    10-element tensors.

    Args:

    * root (str): The root directory of the image files.
    * train (bool): Whether to load training or testing data.
    * download (bool): Whether to download to root if files not present.
    """

    # The following two transform functions are used by torch.utils.data.DataSet
    # to read MNIST data from a binary file and properly transform into image
    # tensors and one-hot coded labels for use by the network.

    # Transforms PIL.Image into a torch.Tensor
    # Any other transforms can be added to the list
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    # Transforms a single label using one-hot coding into a binary array
    # pylint: disable=E0602
    # pylint: disable=E1101
    # pylint: disable=E1102
    # (^ disables incorrect error warning by pylint)
    binarizer = LabelBinarizer().fit(np.arange(10))
    target_transform = lambda x: torch.tensor(x, dtype=torch.long)


    def __init__(self, root: str, train: bool=True, download: bool=True):
        super().__init__(download=download, train=train, root=root,
            transform=self.__class__.transform,
            target_transform=self.__class__.target_transform)