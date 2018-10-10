# `net`

This package defines network architectures to be used by various pipelines.

Currently defined nets:

* `Alexnet`: Modified Alexnet such that it can load pre-trained weights for an output layer of any size (final layer weights are ignored). Also does not backpropagate weights to convolutional layers.
