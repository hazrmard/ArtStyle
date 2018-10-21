# `data`

Contains data loaders that provide a `torch.utils.data.Dataset` instance to be used during learning and evaluation.

## Datasets

### `CroppedData`

Returns `224x224` 3 channel RGB crops of paintings.

### `ResizedData`

Returns `224x224` 3 channel RGB resized paintings.
