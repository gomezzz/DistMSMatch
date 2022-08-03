#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
SAVE_DIR="/mnt/home/meoni/projects/DistMSMatch/notebooks/saved_models"         
NET=unet #Options are efficientnet-lite0, efficientnet-b0, unet.
BATCH_SIZE=16
N_EPOCH=100                    
WEIGHT_DECAY=0.00075
LR=0.01 #0.01
RED='\033[0;31m'
BLACK='\033[0m'
#create save location
NUM_LABELS_USED="50 100 500 1000 2000 3000"
SCALE=0.5

#switch to fixmatch folder for execution
cd ..
echo -e "Using GPU ${RED} $CUDA_VISIBLE_DEVICES ${BLACK}."

for NUM_LABELS in $NUM_LABELS_USED; do #Note: they are the total number of labels, not per class.
    #Remove "echo" to launch the script.
    python train.py --num_labels $NUM_LABELS --wd $WEIGHT_DECAY --lr $LR --batch_size $BATCH_SIZE --scale $SCALE --epoch $N_EPOCH --save_dir $SAVE_DIR --net $NET
    wait
done