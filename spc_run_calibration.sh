#!/bin/sh
#SBATCH --partition=t1standard
#SBATCH --ntasks=288
#SBATCH --tasks-per-node=24

# activate environment
module load lang/Anaconda3/5.3.0
source activate oggm_env_v02

REGNO="4"
ADD_CAL_SWITCH=1

# split glaciers into batches for different nodes
python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES -add_cal=$ADD_CAL_SWITCH

# list rgi_glac_number batch filenames
CHECK_STR="Cal_R${REGNO}_rgi_glac_number_batch"
rgi_fns=$(find $CHECK_STR*)
echo rgi_glac_number filenames:$rgi_fns
# create list
list_rgi_fns=($rgi_fns)
echo first_batch:${list_rgi_fns[0]}

echo partition: $SLURM_JOB_PARTITION
echo num_nodes: $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo num_tasks: $SLURM_NTASKS tasks_node: $SLURM_NTASKS_PER_NODE

#counter=1
#for i in $rgi_fns
#do
#  relative_node=$(( counter % 2))
#  # print the filename
#  echo $i
#  # run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
#  srun -N 1 -n 1 -r $relative_node python run_calibration_woggm.py -num_simultaneous_processes=24 -rgi_glac_number_fn=$i &
#  counter=$(( counter + 1 ))
#done
## wait tells the loop to not move on until all the srun commands are completed
#wait

#for i in $rgi_fns
#do
  # print the filename
#  echo $i
  
  # determine batch number
  #BATCHNO="$(cut -d'.' -f1 <<<$(cut -d'_' -f7 <<<"$i"))"
  #echo $BATCHNO
  
#  # run the file on a separate node (& tells the command to move to the next loop for any empty nodes)
#  #srun -N $BATCHNO -n 1 python run_calibration_woggm.py -num_simultaneous_processes=24 -rgi_glac_number_fn=$i &
#  srun -N 1 -n 1 python run_calibration_woggm.py -num_simultaneous_processes=24 -rgi_glac_number_fn=$i &
#   srun --exclusive -N1 -n1 python run_calibration_woggm.py &
#done
# wait tells the loop to not move on until all the srun commands are completed
#wait

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
  #srun --exclusive -N1 -n1 python print_helloworld.py &
  srun  --exclusive -N1 -n1 python run_calibration_woggm.py -num_simultaneous_processes=24 -rgi_glac_number_fn=$rgi_fn &
  echo $NODE
  echo $count
  ((count++))
done
wait

# ----- THIS WORKS ON DEBUG NODE -----
#count=0
## Launch the application
#for NODE in $NODELIST; do
#  #srun --nodes=1 -w $NODE ./test.sh &
#  rgi_fn=${list_rgi_fns[count]}
#  echo $rgi_fn
#  # ONLY WORKS WITH EXCLUSIVE!
#  srun  --exclusive -N1 -n1 python run_calibration_woggm.py -num_simultaneous_processes=24 -rgi_glac_number_fn=$rgi_fn &
#  echo $NODE
#  echo $count
#  ((count++))
#done
#wait
