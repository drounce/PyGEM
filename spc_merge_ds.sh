#!/bin/sh
#SBATCH --partition=t1small
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS tasks_node: $SLURM_NTASKS_PER_NODE

# gcm list
GCM_NAMES_FP="../Climate_data/cmip5/"
GCM_NAMES_FN="gcm_postprocess_list.txt"
# determine gcm names and rcp scenario
GCM_NAMES_LST="$(< $GCM_NAMES_FP$GCM_NAMES_FN)"

# activate environment
module load lang/Anaconda3/2.5.0
source activate pygem_hpc

# Process the GCMs
for GCM_NAME in $GCM_NAMES_LST; do
  GCM_NAME_NOSPACE="$(echo -e "${GCM_NAME}" | tr -d '[:space:]')"
  echo -e "\n$GCM_NAME"
  # run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
  srun -N 1 -n 1 python merge_ds_spc.py -gcm_name="$GCM_NAME_NOSPACE" -num_simultaneous_processes=$SLURM_NTASKS_PER_NODE &
done
wait

echo -e "\nScript finished"
