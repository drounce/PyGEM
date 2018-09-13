#!/bin/sh
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4

module load lang/Anaconda3/2.5.0
source activate pygem_hpc
python run_calibration.py
