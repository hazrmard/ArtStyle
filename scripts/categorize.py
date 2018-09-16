"""
## Categorize

This script converts a single directory of images with different labels into
a directory of directories where each subdirectory contains images of a single
class. Only creates symbolic links. No copies of images are made. Requires:

    * elevated permissions.
    * an input CSV of the form (order of columns does not matter):

        style   |   filename  
        ----------------------
        style1  |   filename1
        style2  |   filename2
        ...

The output subdirectories contain symbolic links.

Input directory:

```
DIR/
    filename1.jpg
    filename2.jpg
    ...
```

Output directory:

```
OUTDIR/
    style1/
        filename1.jpg
    style2/
        filename2.jpg
    ...
```
"""

import os
import platform
import csv
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Split directory of images by labels.')
    parser.add_argument('input',
                        help='CSV file containing file names and labels.',
                        metavar='I.csv')
    parser.add_argument('imgdir',
                        help='Path to directory containing images.',
                        metavar='IMG/')
    parser.add_argument('outdir',
                        help='Directory where to place categorized images.',
                        default='output',
                        nargs='?',
                        metavar='OUT/')
    
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    avail_files = set(os.listdir(args.imgdir))
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        style_index = headers.index('style')
        fname_index = headers.index('filename')
        for line in reader:
            style = line[style_index]
            fname = line[fname_index]
            if fname in avail_files:
                fullsrc = os.path.join(args.imgdir, fname)
                fulldest = os.path.join(args.outdir, style, fname)
                try:
                    os.symlink(fullsrc, fulldest, False)
                except FileNotFoundError:
                    os.makedirs(os.path.join(args.outdir, style), exist_ok=True)
                    os.symlink(fullsrc, fulldest, False)