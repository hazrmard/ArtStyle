from multiprocessing import Process, Queue, cpu_count
import os
import sys
from argparse import ArgumentParser
from glob import glob
from typing import Tuple
import warnings

from PIL import Image

warnings.simplefilter('ignore', Image.DecompressionBombWarning)

root = os.path.join(os.path.expanduser('~'), 'Downloads', 'paintings')
dest = os.path.join(root, 'processed')

parser = ArgumentParser(description='Generate resized + cropped data sets from\
                        downloaded training and testing images.')
parser.add_argument('--root', default=root, help='Folder containing training and\
                    testing directories.')
parser.add_argument('--dest', default=dest, help='Folder to which processed images\
                    are written.')



def preprocess(q: Queue, dest_dirs: Tuple[str], log: Queue):
    """
    Called by a separate process.

    Args:
    * q: Queue of image filepaths to open. A None element exits process,
    * dest_dirs: A tuple of paths to cropped and resized output directories,
    * log: A Queue to log when an image is done.
    """
    # preprocessing parameters
    resize = 300                # resizing square size
    crop = 300                  # crop square size
    halfsize = int(crop / 2)
    crop_dir, resize_dir = dest_dirs
    
    while True:
        fpath = q.get()
        if fpath is None:
            log.put(None)
            return
        fname = os.path.basename(fpath)

        try:
            # Image processing
            im: Image.Image = Image.open(fpath)
            im.load()
            # remove alpha channel
            if im.mode in ('RGBA', 'P'):
                im = im.convert('RGB')        
            cols, rows = im.size
            cx, cy = int(cols / 2), int(rows / 2)
            rect = (cx - halfsize, cy - halfsize, cx + halfsize, cy + halfsize)
            cropped = im.crop(rect)
            resized = im.resize((resize, resize))
            cropped.save(os.path.join(crop_dir, fname))
            resized.save(os.path.join(resize_dir, fname))
            log.put(1)
        except Exception as e:
            log.put(('{}: {}'.format(fname, str(e))))




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

    for dir in [*crop_dirs, *resize_dirs]:
        try:
            os.makedirs(dir)
        except FileExistsError:
            print('Directory: {} already exists.'.format(dir), file=sys.stderr)
            exit(-1)


    N = cpu_count()

    with open(os.path.join(args.dest, 'logfile.txt'), 'w') as logfile:
        # loop over training and testing sets separately
        for dir, dest_dirs in zip((train_dir, test_dir), zip(crop_dirs, resize_dirs)):
            print('Directory: {}'.format(dir))
            log = Queue()
            Q = Queue()
            for fpath in glob(os.path.join(dir, '*.*')):
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
                    logfile.write('')

            for p in P:
                p.join()

    print('\nDone')
