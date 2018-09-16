# Python scripts

This folder contains the following powershell and python scripts:

* `download.ps1`: download training and test data sets,
* `preprocess.py`: generate resized + cropped images from downloaded images.
* `split.py`: split a CSV describing images into training and test sets,
* `categorize.py`: convert a directory of images into subdirectories by style labels.

See scripts for more details.

The directory structure required by the main code is of the form:

```
ROOT/
    train/
        cropped/
            img1, img2,...
        resized/
            img1, img2,...
    test/
        cropped/
            img1, img2,...
        resized/
            img1, img2,...
```