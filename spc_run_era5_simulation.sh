#!/bin/sh
#SBATCH --partition=t1small
#SBATCH --ntasks=48
#SBATCH --tasks-per-node=24

STARTNO="1"
ENDNO="30000"

# region
REGNO="1"
STARTYR="2000"
ENDYR="2019"
MERGE_SWITCH=0
ORDERED_SWITCH=0

# activate environment
module load lang/Anaconda3/5.3.0
source activate oggm_env_v02

# split glaciers into batches for different nodes
python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES -option_ordered=$ORDERED_SWITCH -startno=$STARTNO -endno=$ENDNO -regno=$REGNO
#python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES -option_ordered=$ORDERED_SWITCH

# list rgi_glac_number batch filenames
CHECK_STR="R${REGNO}_rgi_glac_number_${STARTNO}-${ENDNO}glac_batch"
#CHECK_STR="R${REGNO}_rgi_glac_number_batch"
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


count=0
# Launch the application
for NODE in $NODELIST; do
  #srun --nodes=1 -w $NODE ./test.sh &
  rgi_fn=${list_rgi_fns[count]}
  echo $rgi_fn
  # ONLY WORKS WITH EXCLUSIVE!
  srun --exclusive -N1 -n1 python run_simulation_woggm.py -gcm_startyear=$STARTYR -gcm_endyear=$ENDYR -num_simultaneous_processes=24 -rgi_glac_number_fn=$rgi_fn -option_ordered=$ORDERED_SWITCH &
  #echo $NODE
  #echo $count
  ((count++))
done
wait

echo -e "\nScript finished"
