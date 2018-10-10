# ArtStyle

Classifying art using deep learning. Work in progress.

## Code structure

### `scripts`

Contains scripts to download and pre-process the WikiArt dataset obtained from
[Kaggle Painter by Numbers][1].

### `lib`

Python package containing networks, data loaders etc. for use.

See `MNIST.ipynb` for examples of code usage based on the MNIST image dataset.

## Data structure

The raw data is is the WikiArt dataset downladed from [kaggle][1]. The input raw sub-structure is:

```
train/
test/
```

The downloaded images are preprocessed using `scripts/preprocess.py` and output into a new directory which is eventually used by this repository. The required data structure is:

```
train/
    cropped/
        *.jpg
    resized/
        *.jpg
test/
    cropped/
        *.jpg
    resized/
        *.jpg
```

Additionally, the dataset has two `csv` files:

* `train_info.csv` contains `style` and `filename` fields that describe the image labels for the training set.

* `all_data_info.csv` contains `style` and `filename` fields for training *and* testing sets.

[1]: https://www.kaggle.com/c/painter-by-numbers/data