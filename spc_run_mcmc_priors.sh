#!/bin/sh
#SBATCH --partition=t1small
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1

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
  #srun --nodes=1 -w $NODE ./test.sh &
  rgi_fn=${list_rgi_fns[count]}
  echo $rgi_fn
  # ONLY WORKS WITH EXCLUSIVE!
  srun  --exclusive -N1 -n1 python run_mcmc_priors.py &
done
wait