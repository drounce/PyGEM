#!/bin/sh
#SBATCH --partition=t1standard
#SBATCH --ntasks=576
#SBATCH --tasks-per-node=24

GCM_NAMES_FP="../Climate_data/cmip5/"
#GCM_NAMES_FN="gcm_rcp26_filenames_glaciermip_noMPI-ESM-LR.txt"
GCM_NAMES_FN="gcm_rcp26_filenames_single.txt"
# determine gcm names and rcp scenario
GCM_NAMES_LST="$(< $GCM_NAMES_FP$GCM_NAMES_FN)"
RCP="$(cut -d'_' -f2 <<<"$GCM_NAMES_FN")"

# activate environment
module load lang/Anaconda3/2.5.0
source activate pygem_hpc

# delete previous rgi_glac_number batch filenames
find -name 'rgi_glac_number_batch_*' -exec rm {} \;

# split glaciers into batches for different nodes
python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES

# list rgi_glac_number batch filenames
rgi_fns=$(find rgi_glac_number_batch*)
echo rgi_glac_number filenames:$rgi_fns
# create list
list_rgi_fns=($rgi_fns)
echo first_batch:${list_rgi_fns[0]}

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS tasks_node: $SLURM_NTASKS_PER_NODE

for GCM_NAME in $GCM_NAMES_LST; do
  echo "$GCM_NAME"
  echo "$RCP"
  for i in rgi_glac_number_batch*
  do
    # print the filename
    echo $i
    # determine batch number
    BATCHNO="$(cut -d'.' -f1 <<<$(cut -d'_' -f5 <<<"$i"))"
    echo $BATCHNO
    # run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
    srun -N 1 -n 1 python run_simulation.py -gcm_name=$GCM_NAME -rcp=$RCP -num_simultaneous_processes=$SLURM_NTASKS_PER_NODE -rgi_glac_number_fn=$i -batch_number=$BATCHNO &
  done
  # wait tells the loop to not move on until all the srun commands are completed
  wait
done
wait
