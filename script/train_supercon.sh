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
EPOCHS=10
SCR_TOL=50.0

POS_CD=4
POS_ID=3
NEG_CD=2
NEG_ID=2
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


WEIGHT=$5
if [ -z "$WEIGHT" ]
then
  echo "WEIGHT is empty"
  WEIGHT=0.1
else
  echo "WEIGHT is set"
fi
echo $WEIGHT


source /home/qyan/venvtranspose/bin/activate
cd /home/qyan/TransPose


echo start at `date`

case $TASK in
  "coord")
    case $NET_DEPTH in
      "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} --supercon_weight ${WEIGHT} \
            --sampling_pos_cross_dom ${POS_CD} --sampling_pos_in_dom ${POS_ID} --sampling_neg_cross_dom ${NEG_CD} --sampling_neg_in_dom ${NEG_ID} \
            --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL} \
            --uncertainty --auto_resume --tiny --ckpt_dir ${CKPT_DIR}
        ;;
      "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} --supercon_weight ${WEIGHT} \
            --sampling_pos_cross_dom ${POS_CD} --sampling_pos_in_dom ${POS_ID} --sampling_neg_cross_dom ${NEG_CD} --sampling_neg_in_dom ${NEG_ID} \
            --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL} \
            --uncertainty --auto_resume --ckpt_dir ${CKPT_DIR}
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "depth")
    case $NET_DEPTH in
    "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} --supercon_weight ${WEIGHT} \
            --sampling_pos_cross_dom ${POS_CD} --sampling_pos_in_dom ${POS_ID} --sampling_neg_cross_dom ${NEG_CD} --sampling_neg_in_dom ${NEG_ID} \
            --learningrate ${LR} --epochs ${EPOCHS} \
            --softclamp 5 --hardclamp 10 \
            --uncertainty --auto_resume --tiny --ckpt_dir ${CKPT_DIR}
        ;;
    "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} --supercon_weight ${WEIGHT} \
            --sampling_pos_cross_dom ${POS_CD} --sampling_pos_in_dom ${POS_ID} --sampling_neg_cross_dom ${NEG_CD} --sampling_neg_in_dom ${NEG_ID} \
            --learningrate ${LR} --epochs ${EPOCHS} \
            --softclamp 5 --hardclamp 10 \
            --uncertainty --auto_resume --ckpt_dir ${CKPT_DIR}
        ;;
    *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "normal")
    case $NET_DEPTH in
    "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} --supercon_weight ${WEIGHT} \
            --sampling_pos_cross_dom ${POS_CD} --sampling_pos_in_dom ${POS_ID} --sampling_neg_cross_dom ${NEG_CD} --sampling_neg_in_dom ${NEG_ID} \
            --learningrate ${LR} --epochs ${EPOCHS} \
            --softclamp 5 --hardclamp 10 \
            --uncertainty --auto_resume --tiny --ckpt_dir ${CKPT_DIR}
        ;;
    "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} --supercon_weight ${WEIGHT} \
            --sampling_pos_cross_dom ${POS_CD} --sampling_pos_in_dom ${POS_ID} --sampling_neg_cross_dom ${NEG_CD} --sampling_neg_in_dom ${NEG_ID} \
            --learningrate ${LR} --epochs ${EPOCHS} \
            --softclamp 5 --hardclamp 10 \
            --uncertainty --auto_resume --ckpt_dir ${CKPT_DIR}
        ;;
    *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  *)
    echo "$TASK is not a pre-specified task, do nothing..."
esac

echo finished at `date`
