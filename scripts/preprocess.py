"""
## Preprocessing

This script runs on multiple cores. It reads the downloaded images and runs the
manipulations coded in the `preprocess()` function on each image. Any errors
are stored in a `logfile` in the destination directory.

The input directory sub-structure is:

    train/
    test/

The output directory sub-structure is:

    logfile.txt
    train/
        cropped/
        resized/
    test/
        cropped/
        resized/
"""
from multiprocessing import Process, Queue, cpu_count
import os
import sys
from argparse import ArgumentParser
from glob import glob
from typing import Tuple, List
import warnings

import numpy as np
from PIL import Image

# assuming image source is trusted, removing image size limits
Image.MAX_IMAGE_PIXELS = np.inf
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

root = os.path.join(os.path.expanduser('~'), 'Downloads', 'paintings')
dest = os.path.join(root, 'processed')

parser = ArgumentParser(description='Generate resized + cropped data sets from\
                        downloaded training and testing images.')
parser.add_argument('--root', default=root, help='Folder containing "train" and\
                    "test" directories with images.')
parser.add_argument('--dest', default=dest, help='Folder to which processed images\
                    are written.')
parser.add_argument('--redo_log', action='store_true', default=False,\
                    help='Reprocess+overwrite only files entered in logfile.txt.')



def preprocess(q: Queue, dest_dirs: Tuple[str], log: Queue):
    """
    Called by a separate process.

    Args:
    * q: Queue of image filepaths to open. A None element exits process,
    * dest_dirs: A tuple of paths to cropped and resized output directories,
    * log: A Queue to log when an image is done.
    """
    # preprocessing parameters
    resize = 224                # resizing square size
    crop = 224                  # crop square size
    halfsize = int(crop / 2)
    crop_dir, resize_dir = dest_dirs
    
    while True:
        fpath = q.get()
        if fpath is None:
            log.put(None)
            return
        fname = os.path.basename(fpath)

        try:
            saved_crop = False
            # Image processing
            im = Image.open(fpath)
            im.load()
            # convert image to 3-channel rgb
            if im.mode != 'RGB':
                im = im.convert('RGB')
            # transform image    
            cols, rows = im.size
            cx, cy = int(cols / 2), int(rows / 2)
            rect = (cx - halfsize, cy - halfsize, cx + halfsize, cy + halfsize)
            cropped = im.crop(rect)
            resized = im.resize((resize, resize))
            # write to file
            cropped.save(os.path.join(crop_dir, fname))
            saved_crop = True
            resized.save(os.path.join(resize_dir, fname))
            log.put(1)
        except Exception as e:
            if saved_crop:      # remove cropped image if resized cannot be saved
                os.remove(os.path.join(crop_dir, fname))
            log.put(('{}: {}\n'.format(fname, str(e))))



def get_log_fpaths(dir: str, logs: List[str]) -> List[str]:
    """
    read logfile and return list of filepaths if they exist in dir.
    """
    fpaths = []
    for entry in logs:
        index = entry.find(':')
        if index > 0:
            fname = entry[:index]
            fpath = os.path.join(dir, fname)
            if os.path.isfile(fpath):
                fpaths.append(fpath)
    return fpaths




if __name__ == '__main__':

    args = parser.parse_args()

    train_dir = os.path.join(args.root, 'train')
    test_dir = os.path.join(args.root, 'test')
    
    # make output directories
    train_cropped = os.path.join(args.dest, 'train', 'cropped')
    train_resized = os.path.join(args.dest, 'train', 'resized')
    test_cropped = os.path.join(args.dest, 'test', 'cropped')
    test_resized = os.path.join(args.dest, 'test', 'resized')
    crop_dirs = (train_cropped, test_cropped)
    resize_dirs = (train_resized, test_resized)
    logpath = os.path.join(args.dest, 'logfile.txt')

    for dir in [*crop_dirs, *resize_dirs]:
        try:
            os.makedirs(dir)
        except FileExistsError:
            if not args.redo_log:
                print('Directory: {} already exists.'.format(dir), file=sys.stderr)
                exit(-1)


    N = cpu_count()

    with open(logpath, 'a+') as logfile:

        if args.redo_log:
            logfile.seek(0)
            logs = logfile.read().split('\n')
            logfile.seek(0)

        # loop over training and testing sets separately
        for dir, dest_dirs in zip((train_dir, test_dir), zip(crop_dirs, resize_dirs)):
            print('Directory: {}'.format(dir))
            log = Queue()
            Q = Queue()
            if args.redo_log:
                fpaths = get_log_fpaths(dir, logs)
            else:
                fpaths = glob(os.path.join(dir, '*.*'))
            for fpath in fpaths:
                Q.put(fpath)
            for _ in range(N):
                Q.put(None)

            P = []
            for _ in range(N):
                p = Process(target=preprocess, args=(Q, dest_dirs, log))
                p.start()
                P.append(p)

            finished = 0
            count = 0
            while finished < N:
                status = log.get()
                if status is None:
                    finished += 1
                elif status == 1:
                    count += 1
                    print('{} images pre-processed.\r'.format(count), end='')
                else:
                    print(status)
                    logfile.write(status)

            for p in P:
                p.join()
            
        logfile.truncate()

    print('\nDone')
