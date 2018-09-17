#!/bin/sh
#SBATCH --partition=t1standard
#SBATCH --ntasks=48
#SBATCH --tasks-per-node=24

module load lang/Anaconda3/2.5.0
source activate pygem_hpc
python run_calibration.py -num_simultaneous_processes=48
