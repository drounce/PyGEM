#!/bin/sh
#SBATCH --partition=debug
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24

# activate environment
module load lang/Anaconda3/2.5.0
source activate pygem_hpc

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS tasks_node: $SLURM_NTASKS_PER_NODE

# run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
#srun -N 1 -n 1 python run_calibration_4debrispaper.py -num_simultaneous_processes=24

rgi_fns

for i in 1
do
  # print the filename
  echo $i
  # run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
  srun -N 1 -n 1 python run_calibration_4debrispaper.py -num_simultaneous_processes=24 &
done
# wait tells the loop to not move on until all the srun commands are completed
wait