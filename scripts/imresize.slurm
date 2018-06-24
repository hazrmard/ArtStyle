#!/bin/bash
#SBATCH --mail-user=ibrahim.ahmed@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1           # Number of physical nodes to run on (>1 only for job arrays or if MPI available)
#SBATCH --ntasks=1          # Number of instances of same task to spin up (for MPI)
#SBATCH --cpus-per-task=16  # Number of cores to make available per node (for non-distriuted multiprocessing)
#SBATCH --time=05:00:00
#SBATCH --mem=16G            # RAM per node
#SBATCH --account=cs4269
#SBATCH --output=./jobs/outputs/imresize.out


# load environment
# module load Anaconda3/4.4.0
# source activate defaultpy

# Other Commands
echo "RUNNING"
# Finds list of jpg files in scratch directory,
# then uses xargs to break up file list into chunks of 10,
# to feed to mogrify for batch resizing
# b/c bash cannot handle all 103000 filenames as arguments.
find /scratch/ahmedi/paintings -type f -name '*.jpg' -print0 | xargs -0 -n10 mogrify -path /scratch/ahmedi/paintings256 -resize 256x256\!

