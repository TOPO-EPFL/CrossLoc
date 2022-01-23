#!/bin/bash

LR=1e-4
EPOCHS=1000
SCR_TOL=50.0
BATCH_SIZE=12


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


if [ $REAL_DATA_DOM == "in_place" ]
then
    if [ $DATASET == "urbanscape" ]
    then
      echo "Use more epochs for in_place + urbanscape network finetuning: 150 -> 400"
      EPOCHS=400
    else
      echo "Use more epochs for in_place + naturescape network finetuning: 150 -> 800"
      EPOCHS=800
    fi
fi


if [ $REAL_DATA_DOM == "out_of_place" ]
then
    if [ $DATASET == "urbanscape" ]
    then
      echo "Use more epochs for out_of_place + urbanscape network finetuning: 150 -> 1000"
      EPOCHS=1000
      if [ $TASK == "coord" ]
      then
        echo "Use super 3000 long epochs for Urbanscape OOP coord encoder fine-tuning!"
        EPOCHS=3000
      fi
    else
      echo "Use more epochs for out_of_place + naturescape network finetuning: 150 -> 3000"
      EPOCHS=3000
    fi
fi


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


PROJ_DIR=$(pwd)
SIM_DATA_CHUNK=$(printf "%.2f" ${SIM_DATA_CHUNK})
ENC_PRETRAINED=$PROJ_DIR/weights-clean/encoders-pretraining/${DATASET}/${TASK}/model-sc-${SIM_DATA_CHUNK}.net

SP_SESSION=''
if [ "$SIM_DATA_CHUNK" != "1.00" ];
  then
    echo "Partially pretrained encoders are to be finetuned..."
fi
SP_SESSION='pt'$SIM_DATA_CHUNK

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
          --network_in ${ENC_PRETRAINED}  --session "clean_training_${SP_SESSION}"  --no_lr_scheduling
        ;;
      "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --softclamp 100 --hardclamp 1000 \
          --uncertainty ${UNC} --auto_resume --ckpt_dir ${CKPT_DIR} \
          --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
          --network_in ${ENC_PRETRAINED}  --session "clean_training_${SP_SESSION}"  --no_lr_scheduling
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
            --network_in ${ENC_PRETRAINED}  --session "clean_training_${SP_SESSION}"  --no_lr_scheduling
        ;;
    "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --hardclamp 10 \
            --uncertainty ${UNC} --auto_resume --ckpt_dir ${CKPT_DIR} \
            --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
            --network_in ${ENC_PRETRAINED}  --session "clean_training_${SP_SESSION}"  --no_lr_scheduling
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
            --network_in ${ENC_PRETRAINED}  --session "clean_training_${SP_SESSION}"  --no_lr_scheduling
        ;;
    "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} \
            --learningrate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
            --hardclamp 10 \
            --uncertainty ${UNC} --auto_resume --ckpt_dir ${CKPT_DIR} \
            --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
            --network_in ${ENC_PRETRAINED}  --session "clean_training_${SP_SESSION}"  --no_lr_scheduling
        ;;
    *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "semantics")
  # fewer epochs for semantics, always 30!
  EPOCHS=30
    case $NET_DEPTH in
      "TINY")
        python3 train_single_task.py ${DATASET} --task ${TASK} --fullsize \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --uncertainty ${UNC} --auto_resume --tiny --ckpt_dir ${CKPT_DIR} \
          --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
          --network_in ${ENC_PRETRAINED}  --session "clean_training_${SP_SESSION}"  --no_lr_scheduling
        ;;
      "FULL")
        python3 train_single_task.py ${DATASET} --task ${TASK} --fullsize \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --uncertainty ${UNC} --auto_resume --ckpt_dir ${CKPT_DIR} \
          --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk 0.0 \
          --network_in ${ENC_PRETRAINED}  --session "clean_training_${SP_SESSION}"  --no_lr_scheduling
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  *)
    echo "$TASK is not a pre-specified task, do nothing..."
esac

echo finished at `date`
