"""
Runs a feature extraction function on a single node with multiple cores.
Requires a csv file containing file names and labels.

The csv file must be of the format:

style   |   filename
----------------------
style1  |   filename1
style2  |   filename2
...
"""

import multiprocessing as mp
from argparse import ArgumentParser
import os
import sys
import numpy as np
import csv
from scipy import ndimage
from imageio import imread
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
    while True:
        data = inq.get()
        if data is None:
            outq.put(None)
            break
        else:
            try:
                im = cv2.imread(os.path.join(dir, data), cv2.IMREAD_COLOR)
                if im is not None:
                    result = extract(im)
                    outq.put(result)
                else:
                    raise Exception('Could not read image:{0:20s}'.format(data))
            except Exception as exp:
                outq.put(exp)



def extract(im: np.ndarray):
    im = cv2.resize(im, (128, 128))
    return im.shape



if __name__ == '__main__':
    inq, outq = mp.Queue(), mp.Queue()
    processes = []

    args = parser.parse_args()
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        fnames = [r[1] for r in reader]
    
    for i in (fnames[:100] + ([None] * args.n)):
        inq.put(i)

    for i in range(os.cpu_count()):
        process = mp.Process(target=worker, args=(inq, outq, args.dir))
        processes.append(process)
        process.start()
    
    if args.output != sys.stdout:
        args.output = open(args.output, 'w', encoding='utf-8')
 
    num_none = 0
    i = 0
    N = len(fnames)
    while num_none < os.cpu_count():
        data = outq.get()
        if data is None:
            num_none += 1
        else:
            if ((i % 100 == 0) or (i == N-1)):
                print('{0:6.3f}% done.\r'.format(i * 100 / N), end='', file=sys.stderr)
            if isinstance(data, Exception):
                print(data, file=sys.stderr)
            else:
                print(i, data, file=args.output)
            i += 1
    print()
    
    for process in processes:
        process.join()
    
    if args.output != sys.stdout:
        args.output.close()