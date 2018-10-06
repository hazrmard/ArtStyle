# `pipeline`

A pipeline is a `module` that contains three classes:

* `Net`: a `torch.nn.Module` representing the neural network.

* `Data`: a `torch.utils.data.Dataset` instance which can be indexed to return tuples of training data and labels.

* `Model`: a `models.Model` instance that encapsulates training, evaluation, and serialization logic.

A pipeline serves to compartmentalize different approaches using shared components.
