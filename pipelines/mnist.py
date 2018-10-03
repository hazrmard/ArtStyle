"""
Defines a prototypical model module. It contains:

* `Net (torch.nn.Module)`: A pytorch neural network definition.
* `Data (torch.utils.data.DataSet)`: A pytorch dataset to load instances.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from tqdm.autonotebook import tqdm, trange


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
    * `binarize (bool)`: Whether to one-hot code integer labels. False (default)
    for cross entropy loss.
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
    # pylint: disable=E0602,E1101,E1102
    # (^ disables incorrect error warning by pylint)
    binarizer = LabelBinarizer().fit(np.arange(10))
    target_transform = lambda x: torch.tensor(x, dtype=torch.long)
    target_transform_b = lambda x: torch.tensor(binarizer.transform(x), dtype=torch.long)


    def __init__(self, root: str, train: bool=True, download: bool=True,
        binarize: bool=False):

        super().__init__(download=download, train=train, root=root,
            transform=self.__class__.transform,
            target_transform=self.__class__.target_transform if not binarize \
                            else self.__class__.target_transform_b)



class Model:
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
        optimizer: torch.optim.Optimizer, cuda: bool = True):
        
        self.net = net
        # pylint: disable=E1101
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and cuda)\
                                    else "cpu")
        self.criterion = criterion
        self.optimizer = optimizer


    def train(self, dataset: torch.utils.data.Dataset, batch_size: int = 10,
        epochs: int = 1, verbosity: int = 'auto', **kwargs):
        """
        Train the network using provided hyperparameters.

        Args:

        * `dataset (torch.utils.data.Dataset)`: The dataset object containing instances.
        * `batch_size (int)`: Number of instances per minibatch.
        * `epochs (int)`: Number of times to iterate over dataset.
        * `verbosity (int)`: Minibatches after which to print statistics. If 'auto'
        then prints only 10 updates.
        * `kwargs`: Any keyword arguments to `DataLoader`.
        """
        # get net's current training mode, and set it to train
        curr_mode = self.net.training
        self.net.train(True)

        if verbosity == 'auto':
            verbosity = max(len(dataset) // (batch_size * 10), 1)

        trainloader = DataLoader(dataset, batch_size=batch_size)
        self.net.to(self.device)

        for epoch in trange(epochs):
            tqdm.write('Epoch # {}'.format(epoch))
            running_loss = 0.
            for i, (batchX, batchY) in enumerate(tqdm(trainloader), 1):
                batchX, batchY = batchX.to(self.device), batchY.to(self.device)
                
                self.optimizer.zero_grad()           # clear gradients from prev. step
                predY = self.net(batchX)             # get predicted labels
                loss = self.criterion(predY, batchY) # compute loss
                loss.backward()                      # backpropagate - populate error grad.
                self.optimizer.step()                # update weights based on gradients
                
                running_loss += loss.item()
                if i % verbosity == 0:
                    tqdm.write('Min-batch # {0:4d}\t Loss: {1:3.3f}'.format(i, running_loss / verbosity))
                    running_loss = 0.
        # restore net's training mode
        self.net.train(curr_mode)


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict labels for instances.
        """
        # get net's current training mode, and set it to train
        curr_mode = self.net.training
        self.net.train(False)
        with torch.no_grad():
            return self.net(x)
        self.net.train(curr_mode)


    def evaluate(self, dataset: torch.utils.data.Dataset, batch_size: int = 10,
        topn: int = 1) -> float:
        """
        Evaluate the model using instances in dataset.

        Args:

        * `dataset (torch.utils.data.Dataset)`: The dataset object containing instances.
        * `batch_size (int)`: Number of instances per minibatch.
        * `topn (int)`: Evaluate if top-N predictions per instance have actual label.
        * `kwargs`: Any keyword arguments to `DataLoader`.

        Returns:

        * `accuracy (float)`: The fraction of predictions correct.
        """
        # get net's current training mode, and set it to train
        curr_mode = self.net.training
        self.net.train(False)

        testloader = DataLoader(dataset, batch_size=batch_size)
        self.net.to(self.device)
        # pylint: disable=E1101
        total = 0
        with torch.no_grad():
            for batchX, batchY in tqdm(testloader):
                # convert to GPU tensor
                batchX = batchX.to(self.device)
                # convert to numpy array of batch_size x labels
                # this is to add an explicit column dimension
                batchY = batchY.view(len(batchY), -1).data.numpy()
                # calculate scores for each class as numpy array batch_size x classes
                scores = self.net(batchX).cpu().numpy()
                # get ranked predictions of each class (last most likely)
                pred = np.argsort(scores, axis=1)
                # select top-n predictions batch_size x topn
                topn_pred = pred[:, -topn:]
                correct = np.any(batchY == topn_pred, axis=1)
                total += correct.sum()
        self.net.train(curr_mode)
        return total / len(dataset)


    def save(self, path: str):
        net_state = self.net.state_dict()
        if self.optimizer is not None:
            opt_state = self.optimizer.state_dict()
        else:
            opt_state = {}
        if self.criterion is not None:
            cri_state = self.criterion.state_dict()
        else:
            cri_state = {}
        torch.save((net_state, opt_state, cri_state), path)


    def load(self, path: str):
        net_state, opt_state, cri_state = torch.load(path, map_location='cpu')
        self.net.load_state_dict(net_state)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_state)
        if self.criterion is not None:
            self.criterion.load_state_dict(cri_state)
