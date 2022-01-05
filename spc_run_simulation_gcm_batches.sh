#!/bin/sh
#SBATCH --partition=t1small
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24

STARTNO="1"
ENDNO="30000"

# region
REGNO="19"
STARTYR="2000"
ENDYR="2100"
MERGE_SWITCH=0
ORDERED_SWITCH=0

# GCM list
#GCM_NAMES_LST="CanESM2 CCSM4 CNRM-CM5 CSIRO-Mk3-6-0 GFDL-CM3 GFDL-ESM2M GISS-E2-R IPSL-CM5A-LR MPI-ESM-LR NorESM1-M"
#GCM_NAMES_LST="BCC-CSM2-MR CESM2 CESM2-WACCM EC-Earth3 EC-Earth3-Veg FGOALS-f3-L GFDL-ESM4 INM-CM4-8 INM-CM5-0 MPI-ESM1-2-HR MRI-ESM2-0 NorESM2-MM"
#GCM_NAMES_LST="EC-Earth3 EC-Earth3-Veg GFDL-ESM4 MRI-ESM2-0"
GCM_NAMES_LST="FGOALS-f3-L"

# Scenarios list
#SCENARIOS="rcp26 rcp45 rcp85"
SCENARIOS="ssp126 ssp245 ssp370 ssp585"
#SCENARIOS="ssp119"

# activate environment
module load lang/Anaconda3/5.3.0
source activate oggm_env_v02


# split glaciers into batches for different nodes
python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES -option_ordered=$ORDERED_SWITCH -startno=$STARTNO -endno=$ENDNO -regno=$REGNO
#python spc_split_glaciers.py -n_batches=$SLURM_JOB_NUM_NODES -option_ordered=$ORDERED_SWITCH -startno=$STARTNO -endno=$ENDNO
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


for GCM_NAME in $GCM_NAMES_LST; do
  GCM_NAME_NOSPACE="$(echo -e "${GCM_NAME}" | tr -d '[:space:]')"
  echo -e "\n$GCM_NAME"
  
  for SCENARIO in $SCENARIOS; do
    echo "$SCENARIO"

    count=0
    # Launch the application
    for NODE in $NODELIST; do
      #srun --nodes=1 -w $NODE ./test.sh &
      rgi_fn=${list_rgi_fns[count]}
      echo $rgi_fn
      # ONLY WORKS WITH EXCLUSIVE!
      srun --exclusive -N1 -n1 python run_simulation_woggm.py -gcm_startyear=$STARTYR -gcm_endyear=$ENDYR -gcm_name="$GCM_NAME_NOSPACE" -scenario="$SCENARIO" -num_simultaneous_processes=24 -rgi_glac_number_fn=$rgi_fn -option_ordered=$ORDERED_SWITCH &
      #echo $NODE
      #echo $count
      ((count++))
    done
    wait
  
  done
  wait
  
done
wait

echo -e "\nScript finished"
