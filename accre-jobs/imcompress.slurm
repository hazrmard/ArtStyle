#!/bin/bash
#SBATCH --mail-user=ibrahim.ahmed@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1           # Number of physical nodes to run on (>1 only for job arrays or if MPI available)
#SBATCH --ntasks=1          # Number of instances of same task to spin up (for MPI)
#SBATCH --cpus-per-task=4   # Number of cores to make available per node (for non-distriuted multiprocessing)
#SBATCH --time=03:00:00
#SBATCH --mem=8G            # RAM per node
#SBATCH --account=cs4269
#SBATCH --output=./jobs/outputs/imcompress.out


# load environment
# module load Anaconda3/4.4.0
# source activate defaultpy

# Other Commands
echo "RUNNING"
# compresses folder to tar.gz file for download
tar -czvf /scratch/ahmedi/paintings256.tar.gz /scratch/ahmedi/paintings256
