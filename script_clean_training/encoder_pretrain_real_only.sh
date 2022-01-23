#!/bin/bash
#SBATCH --chdir /home/qyan/TransPose
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --ntasks 1
#SBATCH --account topo
#SBATCH --mem 96G
#SBATCH --time 72:00:00
#SBATCH --partition gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1

LR=1e-4
EPOCHS=2000
SCR_TOL=50.0
BATCH_SIZE=12

# izar-specific arguments
CKPT_DIR=/scratch/izar/$USER/ckpt-crossloc
source /home/qyan/venvcrossloc/bin/activate
cd /home/qyan/TransPose


DATASET=$1
if [ -z "$DATASET" ]
then
  echo "DATASET is empty"
  DATASET="urbanscape"
else
  echo "DATASET is set"
fi
echo $DATASET


TASK=$2
if [ -z "$TASK" ]
then
  echo "TASK is empty"
  TASK=NONE
else
  echo "TASK is set"
fi
echo $TASK


NET_DEPTH=$3
if [ -z "$NET_DEPTH" ]
then
  echo "NET_DEPTH is empty"
  NET_DEPTH=FULL
else
  echo "NET_DEPTH is set"
fi
echo $NET_DEPTH


SIM_DATA_CHUNK=$4
if [ -z "$SIM_DATA_CHUNK" ]
then
  echo "SIM_DATA_CHUNK is empty"
  SIM_DATA_CHUNK=1.0
else
  echo "SIM_DATA_CHUNK is set"
fi
echo $SIM_DATA_CHUNK


REAL_DATA_DOM=$5
if [ -z "$REAL_DATA_DOM" ]
then
  echo "REAL_DATA_DOM is empty"
  REAL_DATA_DOM="in_place"
else
  echo "REAL_DATA_DOM is set"
fi
echo "$REAL_DATA_DOM"


REAL_DATA_CHUNK=$6
if [ -z "$REAL_DATA_CHUNK" ]
then
  echo "REAL_DATA_CHUNK is empty"
  REAL_DATA_CHUNK=1.0
else
  echo "REAL_DATA_CHUNK is set"
fi
echo $REAL_DATA_CHUNK


UNC=$7
if [ -z "$UNC" ]
then
  echo "UNC is empty"
  UNC="none"
else
  echo "UNC is set"
fi
echo $UNC


CUDA_ID=$8
if [ -z "$CUDA_ID" ]
then
  echo "CUDA_ID is empty"
  CUDA_ID=0
else
  echo "CUDA_ID is set"
fi
echo $CUDA_ID
export CUDA_VISIBLE_DEVICES=${CUDA_ID}


echo start at `date`


case $TASK in
  "coord")
    case $NET_DEPTH in
      "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --softclamp 100 --hardclamp 1000 \
          --uncertainty ${UNC} --auto_resume --tiny --ckpt_dir ${CKPT_DIR} \
          --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
          --no_lr_scheduling --real_only --session "clean_training"
        ;;
      "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --softclamp 100 --hardclamp 1000 \
          --uncertainty ${UNC} --auto_resume --ckpt_dir ${CKPT_DIR} \
          --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
          --no_lr_scheduling --real_only --session "clean_training"
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "depth")
    case $NET_DEPTH in
    "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --hardclamp 10 \
            --uncertainty ${UNC} --auto_resume --tiny --ckpt_dir ${CKPT_DIR} \
            --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
            --no_lr_scheduling --real_only --session "clean_training"
        ;;
    "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --hardclamp 10 \
            --uncertainty ${UNC} --auto_resume --ckpt_dir ${CKPT_DIR} \
            --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
            --no_lr_scheduling --real_only --session "clean_training"
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "normal")
    case $NET_DEPTH in
    "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --hardclamp 10 \
            --uncertainty ${UNC} --auto_resume --tiny --ckpt_dir ${CKPT_DIR} \
            --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
            --no_lr_scheduling --real_only --session "clean_training"
        ;;
    "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --hardclamp 10 \
            --uncertainty ${UNC} --auto_resume --ckpt_dir ${CKPT_DIR} \
            --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
            --no_lr_scheduling --real_only --session "clean_training"
        ;;
    *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "semantics")
  # fewer epochs for semantics
  EPOCHS=60
    case $NET_DEPTH in
      "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} --fullsize \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --uncertainty ${UNC} --auto_resume --tiny --ckpt_dir ${CKPT_DIR} \
          --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
          --no_lr_scheduling --real_only --session "clean_training"
        ;;
      "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} --fullsize \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --uncertainty ${UNC} --auto_resume --ckpt_dir ${CKPT_DIR} \
          --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
          --no_lr_scheduling --real_only --session "clean_training"
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  *)
    echo "$TASK is not a pre-specified task, do nothing..."
esac

echo finished at `date`
