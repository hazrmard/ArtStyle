#!/bin/bash
#SBATCH --mail-user=ibrahim.ahmed@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --account=cs4269
#SBATCH --output=./jobs/outputs/extract.out
#SBATCH --error=./jobs/outputs/extract.err

# This job extracts features defined in github.com/hazrmard/ArtStyle -> extractfeatures.py from a directory
# of images of paintings. The job runs on a single node with 12 processes. The final output is a csv with
# no header.

# load environment
module load Anaconda3/4.4.0
source activate defaultpy

# Other Commands
python ArtStyle/extractfeatures.py "/scratch/ahmedi/paintings/" --input "ArtStyle/data/train.csv" --output "ArtStyle/features.csv" -n 16
