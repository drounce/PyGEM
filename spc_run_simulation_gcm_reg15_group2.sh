#!/bin/sh
#SBATCH --partition=t1small
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS tasks_node: $SLURM_NTASKS_PER_NODE

# region
REGNO="15"

# gcm list
GCM_NAMES_FP="../Climate_data/cmip5/"
GCM_NAMES_FN="gcm_rcp26_filenames_important_group2.txt"
# determine gcm names and rcp scenario
GCM_NAMES_LST="$(< $GCM_NAMES_FP$GCM_NAMES_FN)"
RCP="$(cut -d'_' -f2 <<<"$GCM_NAMES_FN")"

# activate environment
module load lang/Anaconda3/2.5.0
source activate pygem_hpc

# region
#REGNO=$(python pygem_input.py)
echo -e "Region: $REGNO\n"
# region batch string
rgi_batch_str="R${REGNO}_rgi_glac_number_batch"

# delete previous rgi_glac_number batch filenames
find -name '${rgi_batch_str}_*' -exec rm {} \;

# split glaciers into batches for different nodes
python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES -spc_region=$REGNO

# list rgi_glac_number batch filenames
rgi_fns=$(find ${rgi_batch_str}*)
echo rgi_glac_number filenames:$rgi_fns
# create list
list_rgi_fns=($rgi_fns)
echo first_batch:${list_rgi_fns[0]}

for GCM_NAME in $GCM_NAMES_LST; do
  GCM_NAME_NOSPACE="$(echo -e "${GCM_NAME}" | tr -d '[:space:]')"
  echo -e "\n$GCM_NAME"
  echo "$RCP"
  for i in $rgi_fns 
  do
    # print the filename
    echo $i
    # determine batch number
    BATCHNO="$(cut -d'.' -f1 <<<$(cut -d'_' -f6 <<<"$i"))"
    #echo $BATCHNO
    # run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
    srun -N 1 -n 1 python run_simulation.py -gcm_name="$GCM_NAME_NOSPACE" -rcp="$RCP" -num_simultaneous_processes=$SLURM_NTASKS_PER_NODE -spc_region=$REGNO -rgi_glac_number_fn=$i -batch_number=$BATCHNO &
  done
  # wait tells the loop to not move on until all the srun commands are completed
  wait
done
wait

echo -e "\nScript finished"
