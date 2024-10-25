import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

from utils.build_model import build_model_biomedclip
from utils.dataloader import get_data_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss import loss_masks
import utils.misc as misc
from utils.misc import setup_logger

from validate import val

def get_args_parser():
    parser = argparse.ArgumentParser('FGCtrl_Clip_Sam', add_help=False)

    parser.add_argument("--output", type=str, required=True,
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--sam_model_type", type=str, default="vit_l", 
                        help="The type of sam model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--model_type", type=str, default="4patches_256", 
                        help="The type of model to load, in ['4patches_256']")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrained/sam_vit_l_0b3195.pth",
                        help="The path to the SAM checkpoint to use for image encoding.")
    parser.add_argument("--model_checkpoint", type=str, default=None, 
                        help="The path to the model checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=10, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=16, type=int)
    parser.add_argument('--batch_size_valid', default=4, type=int)
    parser.add_argument('--model_save_freq', default=1, type=int)
    parser.add_argument('--log_freq', default=100, type=int)  
    parser.add_argument('--logger', default='train.log', type=str)  
    parser.add_argument('--visualize', default=0, type=int)

    return parser.parse_args()


def train(train_data, val_data, model, args, logger):

    logger.info(f"args: {str(args)}\n")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ### --- Step 1: Train or Valid dataset ---
    
    logger.info("--- create training dataloader ---")
    train_dataloaders, train_datasets = create_dataloaders(train_data,
                                                    my_transforms = [
                                                                RandomHFlip(),
                                                                Resize(args.input_size)
                                                                ],
                                                    batch_size = args.batch_size_train,
                                                    training = True,
                                                    dist=False)
    logger.info(f"train dataloader length : {len(train_dataloaders)}")

    logger.info("--- create valid dataloader ---")
    valid_dataloaders, valid_datasets = create_dataloaders(val_data,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False,
                                                          dist=False)
    logger.info(f"{len(valid_dataloaders)} dataloaders created")
    
    ### --- Step 2: Model Setup ---
    
    model.to(device=args.device)
 
    ### --- Step 3: Optimizer ---
    
    logger.info("--- define optimizer ---")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
    lr_scheduler.last_epoch = args.start_epoch
    
    os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    ### --- Step 4: Train epochs ---

    model.train()
    
    # test_stats = val(args, model, valid_dataloaders, logger=logger)
    # train_stats.update(test_stats)
    
    for epoch in range(epoch_start, epoch_num): 
        logger.info(f"epoch:    {epoch} learning rate:  {optimizer.param_groups[0]['lr']} ")
        metric_logger = misc.MetricLogger(delimiter="  ")

        for data in metric_logger.log_every(train_dataloaders, args.log_freq, logger=logger):
            output = model(
                batched_input=data, multimask_output=False
            )
            mask_logits = output["logits"]
            labels = data['label'].to(mask_logits.device)
            
            loss_mask, loss_dice = loss_masks(mask_logits, labels / 255.0, len(mask_logits))
            loss = loss_mask + loss_dice
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss.item(), **loss_dict)

        ### --- Step 5: Update loggers ---

        logger.info(f"Finished epoch:      {epoch}")
        metric_logger.synchronize_between_processes()
        logger.info(f"Averaged stats:  {metric_logger}")
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        
        lr_scheduler.step()
        test_stats = val(args, model, valid_dataloaders, logger=logger)
        train_stats.update(test_stats)
        
        model.train()  

        if epoch % args.model_save_freq == 0:
            model_name = f"/{args.logger.split('.')[0]}_epoch_{str(epoch)}.pth"
            if misc.is_main_process():
                logger.info('model save at', args.output + model_name)
            misc.save_on_master(model.module.state_dict(), args.output + model_name)
    
    logger.info("Training Reaches The Maximum Epoch Number")

if __name__ == "__main__":

    train_annotations = [
        "data/brain_mri_kaggle3m/annotations/train.txt",
        "data/kvasir_seg/annotations/train.txt",
        "data/retinal/annotations/train_4copies.txt",
        "data/busi/annotations/train.txt"
    ]
    val_annotations = [
        "data/brain_mri_kaggle3m/annotations/val.txt",
        "data/kvasir_seg/annotations/val.txt",
        "data/retinal/annotations/val.txt",
        "data/busi/annotations/val.txt"
    ]
    
    args = get_args_parser()
    logger = setup_logger(os.path.join(args.output, args.logger))
    
    train_data = get_data_dict(train_annotations, logger=logger) 
    val_data = get_data_dict(val_annotations, logger=logger)
    
    model = build_model_biomedclip(
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint,
        model_type = args.model_type,
        model_checkpoint=args.model_checkpoint
    )

    train(train_data, val_data, model, args, logger)
