from .cropped import Data as CroppedData


# The column headers in the csv file describing images.
FNAME_COL = 'filename'
LABEL_COL = 'style'


class Data(CroppedData):
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

    subdir = 'resized'
