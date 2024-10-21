import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

from utils.build_model import build_model
from utils.dataloader import get_data_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss import loss_masks
from utils.visualize import show_anns
import utils.misc as misc

def get_args_parser():
    parser = argparse.ArgumentParser('FGCtrl_Clip_Sam', add_help=False)

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--sam_model_type", type=str, default="vit_l", 
                        help="The type of sam model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--model_type", type=str, default="4patches_256", 
                        help="The type of model to load, in ['4patches_256']")
    parser.add_argument("--sam_checkpoint", type=str, required=True, 
                        help="The path to the SAM checkpoint to use for image encoding.")
    parser.add_argument("--checkpoint", type=str, default=None, 
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
    # parser.add_argument('--batch_size_valid', default=4, type=int)
    parser.add_argument('--model_save_freq', default=2, type=int)
    parser.add_argument('--log_freq', default=100, type=int)  

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    # parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--visualize', action='store_true')
    # parser.add_argument("--restore-model", type=str,
    #                     help="The path to model's training checkpoint for evaluation")

    return parser.parse_args()


def train(train_data, val_data, model, args):

    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_dataloaders, train_datasets = create_dataloaders(train_data,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_dataloaders, valid_datasets = create_dataloaders(val_data,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: DistributedDataParallel---
    
    if torch.cuda.is_availabel():
        model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    model_without_ddp = model.module

 
    ### --- Step 3: Optimizer ---
    
    print("--- define optimizer ---")
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
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders,args.log_freq):
            output = model(
                batched_input=data, multimask_output=False
            )
            mask_logits = output["logits"]
            labels = data['label'].to(mask_logits.device())
            
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


        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(args, model, valid_dataloaders)
        train_stats.update(test_stats)
        
        model.train()  

        if epoch % args.model_save_freq == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(model.module.state_dict(), args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and hq_decoder
    if misc.is_main_process():
        sam_ckpt = torch.load(args.checkpoint)
        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            sam_key = 'mask_decoder.'+key
            if sam_key not in sam_ckpt.keys():
                sam_ckpt[sam_key] = hq_decoder[key]
        model_name = "/sam_hq_epoch_"+str(epoch)+".pth"
        torch.save(sam_ckpt, args.output + model_name)



def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)

def evaluate(args, model, valid_dataloaders, visualize=False):
    model.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader,args.log_freq):
        
            with torch.no_grad():
                output = model(
                    batched_input=data_val, multimask_output=False
                )
            mask_logits = output["logits"]
            labels_ori = data_val["label_ori"].to(mask_logits.device())
            iou = compute_iou(mask_logits, labels_ori)
            boundary_iou = compute_boundary_iou(mask_logits, labels_ori)

            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_vis = output["masks"].cpu()
                gt_vis = (F.interpolate(labels_ori.detach(), (1024, 1024)) > 0).cpu()
                
                for ii, img in enumerate(data_val["pil_image"]):
                    base = data_val['imidx'][ii].item()
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    filename = os.path.join(args.output, f"{k}_{base:03d}")
                    show_anns(masks_vis[ii], gt_vis[ii], filename, show_iou, show_boundary_iou)
                    
            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)


        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)


    return test_stats


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

    train_data = get_data_dict(train_annotations) 
    val_data = get_data_dict(val_annotations)
    args = get_args_parser()
    
    model = build_model(
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint,
        model_type = args.model_type,
        model_checkpoint=args.model_checkpoint
    )

    train(train_data, val_data, model, args)
