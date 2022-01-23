#!/bin/bash

MIN_CKPT_ITER=800000
MIN_CKPT_ITER_SE=0
MAX_CKPT_ITER=1e99

SECTION_NM=val_drone_real


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


UNC=$4
if [ -z "$UNC" ]
then
  echo "UNC is empty"
  UNC="none"
else
  echo "UNC is set"
fi
echo $UNC


CUDA_ID=$5
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
        python3 test_single_task.py ${DATASET} --task ${TASK} --uncertainty ${UNC} --section ${SECTION_NM} \
          --network_in ${CKPT_DIR} --tiny --min_ckpt_iter ${MIN_CKPT_ITER} --max_ckpt_iter ${MAX_CKPT_ITER} \
          --keywords ${DATASET} ${TASK} -tiny real_only
        ;;
      "FULL")
        python3 test_single_task.py ${DATASET} --task ${TASK} --uncertainty ${UNC} --section ${SECTION_NM} \
          --network_in ${CKPT_DIR} --min_ckpt_iter ${MIN_CKPT_ITER} --max_ckpt_iter ${MAX_CKPT_ITER} \
          --keywords ${DATASET} ${TASK} real_only
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "depth")
    case $NET_DEPTH in
    "TINY")
        python3 test_single_task.py ${DATASET} --task ${TASK} --uncertainty ${UNC} --section ${SECTION_NM} \
          --network_in ${CKPT_DIR} --tiny --min_ckpt_iter ${MIN_CKPT_ITER} --max_ckpt_iter ${MAX_CKPT_ITER} \
          --keywords ${DATASET} ${TASK} -tiny real_only
        ;;
    "FULL")
        python3 test_single_task.py ${DATASET} --task ${TASK} --uncertainty ${UNC} --section ${SECTION_NM} \
          --network_in ${CKPT_DIR} --min_ckpt_iter ${MIN_CKPT_ITER} --max_ckpt_iter ${MAX_CKPT_ITER} \
          --keywords ${DATASET} ${TASK} real_only
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "normal")
    case $NET_DEPTH in
    "TINY")
        python3 test_single_task.py ${DATASET} --task ${TASK} --uncertainty ${UNC} --section ${SECTION_NM} \
          --network_in ${CKPT_DIR} --tiny --min_ckpt_iter ${MIN_CKPT_ITER} --max_ckpt_iter ${MAX_CKPT_ITER} \
          --keywords ${DATASET} ${TASK} -tiny real_only
        ;;
    "FULL")
        python3 test_single_task.py ${DATASET} --task ${TASK} --uncertainty ${UNC} --section ${SECTION_NM} \
          --network_in ${CKPT_DIR} --min_ckpt_iter ${MIN_CKPT_ITER} --max_ckpt_iter ${MAX_CKPT_ITER} \
          --keywords ${DATASET} ${TASK} real_only
        ;;
    *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  "semantics")
  # note that we have fewer epochs for semantic task
    case $NET_DEPTH in
      "TINY")
        python3 test_single_task.py ${DATASET} --task ${TASK} --uncertainty ${UNC} --fullsize --section ${SECTION_NM} \
          --network_in ${CKPT_DIR} --tiny --min_ckpt_iter ${MIN_CKPT_ITER_SE} --max_ckpt_iter ${MAX_CKPT_ITER} \
          --keywords ${DATASET} ${TASK} -tiny real_only
        ;;
      "FULL")
        python3 test_single_task.py ${DATASET} --task ${TASK} --uncertainty ${UNC} --fullsize --section ${SECTION_NM} \
          --network_in ${CKPT_DIR} --min_ckpt_iter ${MIN_CKPT_ITER_SE} --max_ckpt_iter ${MAX_CKPT_ITER} \
          --keywords ${DATASET} ${TASK} real_only
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  *)
    echo "$TASK is not a pre-specified task, do nothing..."
esac

echo finished at `date`
