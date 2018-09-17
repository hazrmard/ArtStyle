import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelBinarizer
import numpy as np



class Data(Dataset):

    def __init__(self, root: str, train: bool=True):
        if train:
            root = os.path.join(root, 'train', 'cropped')