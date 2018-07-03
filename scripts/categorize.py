"""
This script converts a single directory of images with different labels into
a directory of directories where each subdirectory contains images of a single
class. Requires:
    * elevated permissions.
    * an input CSV of the form (with optional columns to the right):

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
        next(reader)
        for style, fname, *_ in reader:
            if fname in avail_files:
                fullsrc = os.path.join(args.imgdir, fname)
                fulldest = os.path.join(args.outdir, style, fname)
                try:
                    os.symlink(fullsrc, fulldest, False)
                except FileNotFoundError:
                    os.makedirs(os.path.join(args.outdir, style), exist_ok=True)
                    os.symlink(fullsrc, fulldest, False)