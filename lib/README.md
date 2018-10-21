# `lib`

Bulk of python code for learning from dataset.

## `net`

Defines networks  as `torch.nn.Module` subclasses for use in the pipeline.

## `data`

Defines datasets as `torch.utils.data.Dataset` subclasses for use in the pipeline.

## `pipeline`

Each module in the package defines a single learning pipeline:

```
Data    --->
            |-->    Model
Net     --->
```

A pipeline is a compartmentalized approach to the learning problem.
