import os
import csv

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
    """

    subdir = 'cropped'

    def __init__(self, root: str, info_csv: str, train: bool=True, encode: bool=True,
        binarize: bool=False):

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
            binarize=binarize, num_output_channels=3)
        