"""
Runs a feature extraction function on a single node with multiple cores.
Requires a csv file containing file names and labels.

The csv file must be of the format:

style   |   filename
----------------------
style1  |   filename1
style2  |   filename2
...

Outputs a CSV file (with no headers) containing style, filename, and features
calculated in the extract() function.

style1  |   filename1   | feature   |   feature |   feature
style2  |   filename2   | feature   |   feature |   feature
...

usage:

```
>> python extractfeatures.py IMAGE_DIRECTORY -n NUM_PROCESSES --input CSV --output CSV
```

Requires:

* numpy
* scipy
* opencv
"""

import multiprocessing as mp
from argparse import ArgumentParser
import os
import sys
import csv
import numpy as np
from scipy import ndimage
import cv2



parser = ArgumentParser(description='Run multi-core feature extraction function \
                        on image directory.')
parser.add_argument('dir', help='Directory containing ONLY image files.')
parser.add_argument('-n', default=os.cpu_count(), help='Number of processes.', type=int)
parser.add_argument('--input', default='data/train.csv', help='CSV file containing\
                    style labels and filenames.' )
parser.add_argument('--output', help='Output CSV file that contains features + label.',\
                    default=sys.stdout)



def worker(inq: mp.Queue, outq: mp.Queue, dir: str):
    """
    The function executed by each process.

    Args:
    * inq (mp.Queue): A queue containing tuples of (style, filename) to process.
    * outq (mp.Queue): A queue in which function puts a list of features.
    * dir (str): the path where the filenames exist
    """
    while True:
        data = inq.get()
        if data is None:
            outq.put(None)
            break
        else:
            try:
                style, fname = data
                im = cv2.imread(os.path.join(dir, fname), cv2.IMREAD_COLOR)
                if im is not None:
                    result = extract(im)
                    outq.put(data + result)
                else:
                    raise Exception('Could not read image:{0:20s}'.format(fname))
            except Exception as exp:
                outq.put(exp)



def extract(im: np.ndarray):
    """
    The function called by worker on each image. Returns a list of features.

    Args:
    * im (np.ndarray): An array of pixel values.

    Returns:
    * A list of features
    """
    # im = cv2.resize(im, (128, 128))
    numel = np.prod(im.shape)
    mean = ndimage.mean(im)
    std = ndimage.standard_deviation(im)
    histB = list(ndimage.measurements.histogram(im[:,:,0], 0, 255, 32) / numel)
    histG = list(ndimage.measurements.histogram(im[:,:,1], 0, 255, 32) / numel)
    histR = list(ndimage.measurements.histogram(im[:,:,2], 0, 255, 32) / numel)
    return [mean, std] + histB + histG + histR



if __name__ == '__main__':
    inq, outq = mp.Queue(), mp.Queue()
    processes = []

    args = parser.parse_args()
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)
    
    for i in (rows + ([None] * args.n)):
        inq.put(i)

    for i in range(args.n):
        process = mp.Process(target=worker, args=(inq, outq, args.dir))
        processes.append(process)
        process.start()
    
    if args.output != sys.stdout:
        args.output = open(args.output, 'w', encoding='utf-8', newline='')
    writer = csv.writer(args.output)
 
    num_none = 0
    i = 0
    N = len(rows)
    while num_none < args.n:
        data = outq.get()
        if data is None:
            num_none += 1
        else:
            if ((i % 100 == 0) or (i == N-1)):
                print('{0:6.3f}% done.\r'.format(i * 100 / N), end='', file=sys.stderr)
            if isinstance(data, Exception):
                print(data, file=sys.stderr)
            else:
                writer.writerow(data)
                # print(i, data, file=args.output)
            i += 1
    print('\nDone. Joining processes...', file=sys.stderr)
    
    for process in processes:
        process.join()
    
    if args.output != sys.stdout:
        args.output.close()
