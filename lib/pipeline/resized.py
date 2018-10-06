from .cropped import Data as CroppedData
from .model import Model


# The column headers in the csv file describing images.
FNAME_COL = 'filename'
LABEL_COL = 'style'


class Data(CroppedData):

    subdir = 'resized'
