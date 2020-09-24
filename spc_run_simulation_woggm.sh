#!/bin/sh
#SBATCH --partition=debug
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS tasks_node: $SLURM_NTASKS_PER_NODE

MERGE_SWITCH=0
ORDERED_SWITCH=1

# activate environment
module load lang/Anaconda3/2.5.0
source activate pygem_hpc_woggm

# region
REGNO=$(python pygem/pygem_input.py)
echo -e "Region: $REGNO\n"
# region batch string
rgi_batch_str="R${REGNO}_rgi_glac_number_batch"

# delete previous rgi_glac_number batch filenames
find -name '${rgi_batch_str}_*' -exec rm {} \;

# split glaciers into batches for different nodes
python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES -option_ordered=$ORDERED_SWITCH

# list rgi_glac_number batch filenames
rgi_fns=$(find ${rgi_batch_str}*)
echo rgi_glac_number filenames:$rgi_fns
# create list
list_rgi_fns=($rgi_fns)
echo first_batch:${list_rgi_fns[0]}


for i in $rgi_fns 
do
  # print the filename
  echo $i
  
  # determine batch number
  BATCHNO="$(cut -d'.' -f1 <<<$(cut -d'_' -f6 <<<"$i"))"
  echo $BATCHNO
  
  # run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
  srun -N 1 -n 1 python run_simulation_woggm.py -num_simultaneous_processes=24 -rgi_glac_number_fn=$i -batch_number=$BATCHNO&
done
# wait tells the loop to not move on until all the srun commands are completed
wait

#GCM_NAME_NOSPACE="ERA5"

#set batman_list = 1  
# Merge simulation files 
#for batman in batman_list; do  
  # run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
#  srun -N 1 -n 1 python merge_ds_spc.py -gcm_name="$GCM_NAME_NOSPACE" -num_simultaneous_processes=$SLURM_NTASKS_PER_NODE &
  #srun -N 1 -n 1 python run_postprocessing.py -gcm_name="$GCM_NAME_NOSPACE" -rcp="$RCP" -merge_batches=$MERGE_SWITCH
#done
#wait

echo -e "\nScript finished"
