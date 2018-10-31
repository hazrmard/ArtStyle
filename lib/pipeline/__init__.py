
"""
Defines a learning pipeline. Each module contains three classes:
* `Net`: a `torch.nn.Module` representing the neural network.
* `Data`: a `torch.utils.data.Dataset` instance which can be indexed to return
tuples of training data and labels.
* `Model`: a `models.Model` instance that encapsulates training, evaluation, and
serialization logic.
"""
