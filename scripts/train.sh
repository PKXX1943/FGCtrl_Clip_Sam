#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 

python train_dist.py \
    --output "output/4patches_256/biomed_clip/exp2" \
    --sam_model_type "vit_l" \
    --model_type "4patches_256" \
    --sam_checkpoint "pretrained/sam_vit_l_0b3195.pth" \
    --learning_rate 1e-3 \
    --start_epoch 0 \
    --lr_drop_epoch 10 \
    --max_epoch_num 200 \
    --batch_size_train 16 \
    --batch_size_valid 4 \
    --model_save_freq 5 \
    --log_freq 10 \
    --logger "train.log" \
    --seed 42 \
