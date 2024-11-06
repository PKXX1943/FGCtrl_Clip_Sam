import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

from utils.build_model import build_model_biomedclip, build_model_laion_clip
from utils.dataloader import get_data_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss import loss_masks
import utils.misc as misc
from utils.misc import setup_logger

from validate import val

torch.autograd.set_detect_anomaly(True)

def get_args_parser():
    parser = argparse.ArgumentParser('FGCtrl_Clip_Sam', add_help=False)

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset: ['Med', 'ADE20K']")
    parser.add_argument("--sam_model_type", type=str, default="vit_l", 
                        help="The type of sam model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--model_type", type=str, default="4patches_256", 
                        help="The type of model to load, in ['4patches_256']")
    parser.add_argument("--clip", type=str, default="biomed_clip", 
                        help="The type of clip model to load, in ['biomed_clip', 'laion_clip']")
    parser.add_argument("--sam_checkpoint", type=str, required=True, 
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
    parser.add_argument('--model_save_freq', default=2, type=int)
    parser.add_argument('--log_freq', default=100, type=int)  
    parser.add_argument('--logger', default='train.log', type=str)  

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--visualize', default=0, type=int)
    
    parser.add_argument('--similarities', action='store_true')

    return parser.parse_args()


def train(train_data, val_data, model, args, logger):
    
    misc.init_distributed_mode(args)
    if misc.is_main_process():
        logger.info('world size: {}'.format(args.world_size))
        logger.info('rank: {}'.format(args.rank))
        logger.info('local_rank: {}'.format(args.local_rank))
        logger.info(f"args: {str(args)}\n")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    

    ### --- Step 1: Train or Valid dataset ---
    if misc.is_main_process():
        logger.info("--- create training dataloader ---")
    train_dataloaders, train_datasets = create_dataloaders(train_data,
                                                    my_transforms = [
                                                                RandomHFlip(),
                                                                Resize(args.input_size)
                                                                ],
                                                    batch_size=args.batch_size_train,
                                                    training=True,
                                                    )
    if misc.is_main_process():
        logger.info(f"train dataloader length: {len(train_dataloaders)}")
        logger.info("--- create valid dataloader ---")
    valid_dataloaders, valid_datasets = create_dataloaders(val_data,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False,
                                                        )
    if misc.is_main_process():
        logger.info(f"{len(valid_dataloaders)}valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    
    if torch.cuda.is_available():
        model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    model_without_ddp = model.module

 
    ### --- Step 3: Optimizer ---
    if misc.is_main_process():
        logger.info("--- define optimizer ---")
    optimizer = optim.Adam(model_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
    lr_scheduler.last_epoch = args.start_epoch
    
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    ### --- Step 4: Train epochs ---

    model.train()
    _ = model.to(device=args.device)
    
    for epoch in range(epoch_start, epoch_num): 
        if misc.is_main_process():
            logger.info(f"epoch:   {epoch}  learning rate:  {optimizer.param_groups[0]['lr']}")
            
        # if epoch == epoch_start:
        #     logger.info(f"Validate before the first epoch ...")
        #     test_stats = val(args, model, valid_dataloaders, logger=logger)
        
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
        
        for data in metric_logger.log_every(train_dataloaders, args.log_freq, logger=logger, is_main_proc=misc.is_main_process()):
            output = model(
                batched_input=data, similarities=args.similarities, multimask_output=False
            )
            mask_logits = output["logits"]
            labels = data['label'].to(mask_logits.device)
            
            loss_mask, loss_dice = loss_masks(mask_logits, labels/255.0, len(mask_logits))
            loss = loss_mask + loss_dice
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

        ### --- Step 5: Update loggers ---
        if misc.is_main_process():
            logger.info(f"Finished epoch: {epoch}")
        metric_logger.synchronize_between_processes()
        if misc.is_main_process():
            logger.info(f"Averaged stats: {metric_logger}")
            logger.info("\n\n\n")
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = val(args, model, valid_dataloaders, logger=logger)
        train_stats.update(test_stats)
        
        model.train()  

        if epoch % args.model_save_freq == 0:
            model_name = f"/{args.logger.split('.')[0]}_epoch_{str(epoch)}.pth"
            if misc.is_main_process():
                logger.info(f'model save at {args.output + model_name}')
            misc.save_on_master(model.module.state_dict(), args.output + model_name)
    
    # Finish training
    if misc.is_main_process():
        logger.info("Training Reaches The Maximum Epoch Number")
    


if __name__ == "__main__":
    args = get_args_parser()
    logger = setup_logger(os.path.join(args.output, args.logger))
    
    if args.dataset == 'Med':
        train_annotations = [
            "data/brain_mri_kaggle3m/annotations/train_clear.txt",
            "data/kvasir_seg/annotations/train_clear.txt",
            "data/retinal/annotations/train_4copies.txt",
            "data/busi/annotations/train_clear.txt"
        ]
        val_annotations = [
            "data/brain_mri_kaggle3m/annotations/val_clear.txt",
            "data/kvasir_seg/annotations/val_clear.txt",
            "data/retinal/annotations/val_clear.txt",
            "data/busi/annotations/val_clear.txt"
        ]
    elif args.dataset == 'ADE20K':
        train_annotations = [
            "prepared/train.txt"
        ]
        val_annotations = [
            "prepared/val.txt"
        ]
    else:
        raise NotImplementedError
    
    train_data = get_data_dict(train_annotations, logger=logger) 
    val_data = get_data_dict(val_annotations, logger=logger)
    
    if args.clip == 'biomed_clip':
        model = build_model_biomedclip(
            sam_model_type=args.sam_model_type,
            sam_checkpoint=args.sam_checkpoint,
            model_type = args.model_type,
            model_checkpoint=args.model_checkpoint
        )
    elif args.clip == 'laion_clip':
        model = build_model_laion_clip(
            sam_model_type=args.sam_model_type,
            sam_checkpoint=args.sam_checkpoint,
            model_type = args.model_type,
            model_checkpoint=args.model_checkpoint
        )
    else:
        raise NotImplementedError
    train(train_data, val_data, model, args, logger=logger)
