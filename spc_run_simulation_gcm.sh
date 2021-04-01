#!/bin/sh
#SBATCH --partition=t1small
#SBATCH --ntasks=48
#SBATCH --tasks-per-node=24

# region
REGNO="1"
MERGE_SWITCH=0
ORDERED_SWITCH=0

# gcm list
GCM_NAMES_FP="../climate_data/cmip5/"
GCM_NAMES_FN="gcm_rcp26_filenames.txt"
# determine gcm names and rcp scenario
GCM_NAMES_LST="$(< $GCM_NAMES_FP$GCM_NAMES_FN)"
RCP="$(cut -d'_' -f2 <<<"$GCM_NAMES_FN")"


# activate environment
module load lang/Anaconda3/5.3.0
source activate oggm_env_v02


# split glaciers into batches for different nodes
python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES -option_ordered=$ORDERED_SWITCH

# list rgi_glac_number batch filenames
CHECK_STR="R${REGNO}_rgi_glac_number_batch"
rgi_fns=$(find $CHECK_STR*)
echo rgi_glac_number filenames:$rgi_fns
# create list
list_rgi_fns=($rgi_fns)
echo first_batch:${list_rgi_fns[0]}

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS tasks_node: $SLURM_NTASKS_PER_NODE



cd $SLURM_SUBMIT_DIR
# Generate a list of allocated nodes
NODELIST=`srun -l /bin/hostname | awk '{print $2}' | sort -u`
echo $NODELIST


for GCM_NAME in $GCM_NAMES_LST; do
  GCM_NAME_NOSPACE="$(echo -e "${GCM_NAME}" | tr -d '[:space:]')"
  echo -e "\n$GCM_NAME"
  echo "$RCP"

  count=0
  # Launch the application
  for NODE in $NODELIST; do
    #srun --nodes=1 -w $NODE ./test.sh &
    rgi_fn=${list_rgi_fns[count]}
    echo $rgi_fn
    # ONLY WORKS WITH EXCLUSIVE!
    srun --exclusive -N1 -n1 python run_simulation_woggm.py -gcm_name="$GCM_NAME_NOSPACE" -rcp="$RCP" -num_simultaneous_processes=24 -rgi_glac_number_fn=$rgi_fn -option_ordered=$ORDERED_SWITCH &
    echo $NODE
    echo $count
    ((count++))
  done
  wait

#  set batman_list = 1  
#  # Merge simulation files 
#  for batman in batman_list; do
#    # run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
#    srun -N 1 -n 1 python merge_ds_spc.py -gcm_name="$GCM_NAME_NOSPACE" -rcp="$RCP" -num_simultaneous_processes=$SLURM_NTASKS_PER_NODE &
#    #srun -N 1 -n 1 python run_postprocessing.py -gcm_name="$GCM_NAME_NOSPACE" -rcp="$RCP" -merge_batches=$MERGE_SWITCH
#  done
#  wait
  
done
wait

echo -e "\nScript finished"
