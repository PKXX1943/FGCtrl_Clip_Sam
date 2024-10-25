import os
import argparse
import torch
import torch.nn.functional as F
import random
import time
from utils.build_model import build_model_biomedclip
from utils.dataloader import get_data_dict, create_dataloaders,Resize
from utils.visualize import show_anns
import utils.misc as misc
from utils.misc import setup_logger


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
    parser.add_argument("--model_checkpoint", type=str, default=None, 
                        help="The path to the model checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--learning_rate', default=1e-3, type=float)
    # parser.add_argument('--start_epoch', default=0, type=int)
    # parser.add_argument('--lr_drop_epoch', default=10, type=int)
    # parser.add_argument('--max_epoch_num', default=10, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    # parser.add_argument('--batch_size_train', default=16, type=int)
    parser.add_argument('--batch_size_valid', default=4, type=int)
    # parser.add_argument('--model_save_freq', default=2, type=int)
    parser.add_argument('--log_freq', default=100, type=int)  
    parser.add_argument('--logger', default='train.log', type=str)

    parser.add_argument('--visualize', default=0, type=int)
    parser.add_argument("--restore-model", type=str,
                        help="The path to model's training checkpoint for evaluation")

    return parser.parse_args()

def val(args, model, valid_dataloaders, logger=None):
    model.eval()
    if misc.is_main_process():
        logger.info("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        if misc.is_main_process():
            logger.info(f'valid_dataloader[{k}] len:{len(valid_dataloader)}')

        for data_val in (metric_logger.log_every(valid_dataloader, args.log_freq, logger=logger, is_main_proc=misc.is_main_process())):
        
            with torch.no_grad():
                output = model(
                    batched_input=data_val, multimask_output=False
                )
            mask_logits = output["logits"]
            labels_ori = data_val["label_ori"]
            iou = compute_iou(mask_logits, labels_ori)
            dice = compute_dice(mask_logits, labels_ori)

            if args.visualize > 0:
                if misc.is_main_process():
                    logger.info(f"visualize")
                vis_num = 0
                masks_vis = output["masks"].cpu()
                gt_vis = torch.stack([(F.interpolate(label_ori.detach().unsqueeze(0), (1024, 1024)) > 0).cpu() for label_ori in labels_ori])
                
                for ii, img in enumerate(data_val["pil_image"]):
                    vis_num += 1
                    if vis_num > args.visualize:
                        break
                    base = data_val['imidx'][ii].item()
                    show_iou = torch.tensor([iou.item()])
                    show_dice = torch.tensor([dice.item()])
                    outdir = f"{args.output}/visualize"
                    if misc.is_main_process() and not os.path.exists(outdir):
                        os.makedirs(outdir)
                    elif not os.path.exists(outdir):
                        time.sleep(1)
                    filename = os.path.join(outdir, f"dataset{k}_{base:03d}.jpg")
                    show_anns(masks_vis[ii], gt_vis[ii], filename, img, show_iou, show_dice)
                    
            loss_dict = {"val_iou_"+str(k): iou, "val_dice_"+str(k): dice}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)

        if misc.is_main_process():
            logger.info('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if misc.is_main_process():
            logger.info(f"Averaged stats:{metric_logger}")
            logger.info("\n\n\n")
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)


    return test_stats

def compute_iou(preds, targets):
    iou = 0
    for i in range(len(targets)):
        pred = preds[i, :, :, :].unsqueeze(0)
        if len(targets[i].shape) == 2:
            target = targets[i].unsqueeze(0).to(preds.device)
        else:
            target = targets[i].to(preds.device)
        if(pred.shape[2]!=target.shape[2] or pred.shape[1]!=target.shape[1]):
            postprocess_pred = F.interpolate(pred, size=target.size()[1:], mode='bilinear', align_corners=False)
        else:
            postprocess_pred = pred
        iou = iou + misc.mask_iou(postprocess_pred.squeeze(0), target)
    return iou / len(preds)

def compute_dice(preds, targets):
    dice = 0
    for i in range(len(targets)):
        pred = preds[i, :, :, :].unsqueeze(0)
        if len(targets[i].shape) == 2:
            target = targets[i].unsqueeze(0).to(preds.device)
        else:
            target = targets[i].to(preds.device)
        if(pred.shape[2]!=target.shape[2] or pred.shape[1]!=target.shape[1]):
            postprocess_pred = F.interpolate(pred, size=target.size()[1:], mode='bilinear', align_corners=False)
        else:
            postprocess_pred = pred
        dice = dice + misc.mask_dice(postprocess_pred.squeeze(0), target)
    return dice / len(preds)

if __name__ == "__main__":
    args = get_args_parser()
    logger = setup_logger(os.path.join(args.output, args.logger))

    val_annotations = [
        "data/brain_mri_kaggle3m/annotations/val.txt",
        "data/kvasir_seg/annotations/val.txt",
        "data/retinal/annotations/val.txt",
        "data/busi/annotations/val.txt"
    ]

    val_data = get_data_dict(val_annotations, logger=logger)
    valid_dataloaders, valid_datasets = create_dataloaders(
        val_data,
        my_transforms = [Resize(args.input_size)],
        batch_size=args.batch_size_valid,
        training=False,
        dist=False
    )
    
    model = build_model_biomedclip(
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint,
        model_type = args.model_type,
        
        model_checkpoint=args.model_checkpoint
    )

    val(args, model, valid_dataloaders, logger=logger)
