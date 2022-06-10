#!/bin/sh
#SBATCH --partition=analysis
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=214G

# activate environment
module load lang/Anaconda3/5.3.0
source activate oggm_env_v02

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS tasks_node: $SLURM_NTASKS_PER_NODE

cd $SLURM_SUBMIT_DIR
# Generate a list of allocated nodes
NODELIST=`srun -l /bin/hostname | awk '{print $2}' | sort -u`
echo $NODELIST

# Launch the application
for NODE in $NODELIST; do
  srun  --exclusive -N1 -n1 python find_glaciers_2redo.py &
  echo $NODE
done
wait