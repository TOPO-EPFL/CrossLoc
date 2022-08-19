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

LR=2e-4
EPOCHS=151
SCR_TOL=50.0

BATCH_SIZE=4
CKPT_DIR=/scratch/izar/$USER/ckpt-transpose


DATASET=$1
if [ -z "$DATASET" ]
then
  echo "DATASET is empty"
  DATASET=EPFL
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


CUDA_ID=$4
if [ -z "$CUDA_ID" ]
then
  echo "CUDA_ID is empty"
  CUDA_ID=0
else
  echo "CUDA_ID is set"
fi
echo $CUDA_ID
export CUDA_VISIBLE_DEVICES=${CUDA_ID}


source /home/qyan/venvtranspose/bin/activate
cd /home/qyan/TransPose


echo start at `date`

case $TASK in
  "coord")
    case $NET_DEPTH in
      "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} --synthetic_only --supercon_weight 0.0 \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --uncertainty --fullsize --auto_resume --tiny --ckpt_dir ${CKPT_DIR}
        ;;
      "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} --synthetic_only --supercon_weight 0.0 \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --uncertainty --fullsize --auto_resume --ckpt_dir ${CKPT_DIR}
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "depth")
    case $NET_DEPTH in
      "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} --synthetic_only --supercon_weight 0.0 \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --softclamp 5 --hardclamp 10 \
            --uncertainty --fullsize --auto_resume --tiny --ckpt_dir ${CKPT_DIR}
        ;;
      "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} --synthetic_only --supercon_weight 0.0 \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --softclamp 5 --hardclamp 10 \
            --uncertainty --fullsize --auto_resume --ckpt_dir ${CKPT_DIR}
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "normal")
    case $NET_DEPTH in
    "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} --synthetic_only --supercon_weight 0.0 \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --softclamp 5 --hardclamp 10 \
            --uncertainty --fullsize --auto_resume --tiny --ckpt_dir ${CKPT_DIR}
        ;;
    "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} --synthetic_only --supercon_weight 0.0 \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --softclamp 5 --hardclamp 10 \
            --uncertainty --fullsize --auto_resume --ckpt_dir ${CKPT_DIR}
        ;;
    *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  *)
    echo "$TASK is not a pre-specified task, do nothing..."
esac

echo finished at `date`
