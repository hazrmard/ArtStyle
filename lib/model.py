"""
Defines the `Model` class which encapsulates training, evaluation, and
serialization of networks. A `Model` is instantiated with a network, loss funcrtion
(criterion), and optimizer.

The class provides the following functions:

* `train()`: Train on a provided dataset.
* `evaluate()`: Measure accuracy on provided dataset.
* `predict()`: Output scores of network on given minibatch.
* `save()/load()`: Save network parameters to file.
"""
import sys
from os.path import abspath, expanduser, expandvars

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm.autonotebook import tqdm, trange



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
        optimizer: torch.optim.Optimizer, device: str = 'cuda:0'):
        
        self.net = net
        # pylint: disable=E1101
        self.device = torch.device(device)
        self.criterion = criterion
        self.optimizer = optimizer


    def train(self, dataset: torch.utils.data.Dataset, batch_size: int = 10,
        epochs: int = 1, xvalidate: float = 0., verbosity: int = 'auto',
        topn: int = 1, **kwargs):
        """
        Train the network using provided hyperparameters.

        Args:

        * `dataset (torch.utils.data.Dataset)`: The dataset containing instances.
        * `batch_size (int)`: Number of instances per minibatch.
        * `epochs (int)`: Number of times to iterate over dataset.
        * `xvalidate (int)`: Fraction of data set to randomly set aside for
        validation at each epoch. Defaults to 0.
        * `verbosity (int)`: Minibatches after which to print loss. If 'auto'
        then prints only 10 updates.
        * `topn (int)`: top-N accuracy to report for validation set. Default 1.
        * `kwargs`: Any keyword arguments to `DataLoader`.
        """
        # get net's current training mode, and set it to train
        curr_mode = self.net.training
        self.net.train(True)
        self.net.to(self.device)

        N = len(dataset)
        N_val = int(N * xvalidate)
        N_train = N - N_val

        if verbosity == 'auto':
            verbosity = max(N_train // (batch_size * 10), 1)
        elif verbosity is None:
            verbosity = N    # loss messages are never printed


        for epoch in trange(epochs, desc='Epochs', leave=True):

            trainset, valset = random_split(dataset, (N_train, N_val))
            trainloader = DataLoader(trainset, batch_size=batch_size)

            running_loss = 0.
            for i, (batchX, batchY) in enumerate(
                tqdm(trainloader, desc='Epoch {}'.format(epoch + 1), leave=False), 1):

                batchX, batchY = batchX.to(self.device), batchY.to(self.device)

                self.optimizer.zero_grad()           # clear gradients from prev. step
                predY = self.net(batchX)             # get predicted labels
                loss = self.criterion(predY, batchY) # compute loss
                loss.backward()                      # backpropagate - populate error grad.
                self.optimizer.step()                # update weights based on gradients

                running_loss += loss.item()
                if i % verbosity == 0:
                    avg_loss = running_loss / verbosity
                    tqdm.write('Mini-batch # {0:4d}\t Loss: {1:3.3f}'.format(i,
                               avg_loss), file=sys.stderr)
                    running_loss = 0.

            if N_val:
                accuracy = self.evaluate(dataset=valset, batch_size=batch_size, topn=topn)
                tqdm.write('Epoch {0:4d}\t Accuracy: {1:.3f}'.format(epoch + 1,
                           accuracy), file=sys.stderr)

        # restore net's training mode
        self.net.train(curr_mode)


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict label scores for instances.
        """
        # get net's current training mode, and set it to train
        curr_mode = self.net.training
        with torch.no_grad():
            self.net.train(False)
            pred = self.net(x)
            self.net.train(curr_mode)
            return pred


    def evaluate(self, dataset: torch.utils.data.Dataset, batch_size: int = 10,
        topn: int = 1) -> float:
        """
        Evaluate the model as classifier using instances in dataset.

        Args:

        * `dataset (torch.utils.data.Dataset)`: The dataset containing instances.
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
            for batchX, batchY in tqdm(testloader, desc='Evaluating', leave=False):
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
        """
        Saves network weights and hyperparameters in a pickle file.
        """
        path = abspath(expandvars(expanduser(path)))
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
        """
        Loads network weights and hyperparameters from a pickle file. The model
        has to be instantiated with the same `net`, `criterion`, and `optimizer`
        arguments.
        """
        path = abspath(expandvars(expanduser(path)))
        net_state, opt_state, cri_state = torch.load(path, map_location='cpu')
        self.net.load_state_dict(net_state)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_state)
        if self.criterion is not None:
            self.criterion.load_state_dict(cri_state)
