#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH 

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    train_dist.py \
    --output "output/4patches_256/biomed_clip/exp4" \
    --dataset "Med" \
    --sam_model_type "vit_l" \
    --model_type "4patches_256" \
    --clip "biomed_clip" \
    --sam_checkpoint "pretrained/sam_vit_l_0b3195.pth" \
    --learning_rate 1e-3 \
    --start_epoch 0 \
    --lr_drop_epoch 10 \
    --max_epoch_num 200 \
    --batch_size_train 12 \
    --batch_size_valid 4 \
    --model_save_freq 4 \
    --log_freq 10 \
    --seed 42 \
    --similarities \
    --find_unused_params \
    --visualize 1 \
    # --model_checkpoint "output/4patches_256/biomed_clip/exp2_invalid/train_epoch_15.pth" \
