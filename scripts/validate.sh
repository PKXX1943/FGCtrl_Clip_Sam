#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 

python validate.py \
    --output "output/4patches_256/biomed_clip/exp2" \
    --sam_model_type "vit_l" \
    --model_type "4patches_256" \
    --sam_checkpoint "pretrained/sam_vit_l_0b3195.pth" \
    --model_checkpoint "output/4patches_256/biomed_clip/exp2/train_epoch_15.pth" \
    --batch_size_valid 4 \
    --log_freq 10 \
    --logger "val.log" \
    --seed 42 \
    --similarities \
    --visualize 2
