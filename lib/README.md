# `lib`

Bulk of python code for learning from dataset.

## `net`

Defines networks  as `torch.nn.Module` subclasses for use in the pipeline.

## `pipeline`

Each module in the package defines a single learning pipeline:

```
Data    --->
            |-->    Model
Net     --->
```
