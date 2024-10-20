from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import cv2
from PIL import Image
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

def get_data_dict(annotations, shuffle=False):
    print("------------------------------Collecting Datasets--------------------------------")
    data_dict = []
    for i, annotation in enumerate(annotations):
        with open(annotation, 'r') as f:
            dataset_name = f.readline().strip()
            print("--->>>", " Dataset[",i,"] ", dataset_name," <<<---")
            for line in f.readlines():
                tmp_im_list, tmp_gt_list, tmp_caption_list = [], [], []
                tmp_im_list.append(line.split(' | ')[0])
                tmp_gt_list.append(line.split(' | ')[1])
                tmp_caption_list.append(line.split(' ')[2].strip())
            if shuffle:
                combined = list(zip(tmp_im_list, tmp_gt_list, tmp_caption_list))
                random.shuffle(combined)
                shuffled1, shuffled2, shuffled3 = zip(*combined)
                tmp_im_list = list(shuffled1)
                tmp_gt_list = list(shuffled2)
                tmp_caption_list = list(shuffled3)
            data_dict.append({"dataset_name": dataset_name,
                                "img_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "caption":tmp_caption_list})

    return data_dict

def create_dataloaders(data_dict, my_transforms=[], batch_size=1, training=False):
    my_dataloaders = []
    my_datasets = []

    if(len(data_dict)==0):
        return my_dataloaders, my_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8


    if training:
        for i in range(len(data_dict)):   
            my_dataset = MyDataset([data_dict[i]], transform = transforms.Compose(my_transforms))
            my_datasets.append(my_dataset)

        my_dataset = ConcatDataset(my_datasets)
        sampler = DistributedSampler(my_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(my_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        my_dataloaders = dataloader
        my_datasets = my_dataset

    else:
        for i in range(len(data_dict)):   
            my_dataset = MyDataset([data_dict[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
            sampler = DistributedSampler(my_dataset, shuffle=False)
            dataloader = DataLoader(my_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            my_dataloaders.append(dataloader)
            my_datasets.append(my_dataset)

    return my_dataloaders, my_datasets

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class Resize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}

class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()
        
        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        
        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])}






class MyDataset(Dataset):
    def __init__(self, data_dict, transform=None):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        img_path_list = [] # im path
        gt_path_list = [] # gt path
        caption_list = []
        for i in range(0,len(data_dict)):
            dataset_names.append(data_dict[i]["dataset_name"])
            dt_name_list.extend([data_dict[i]["dataset_name"] for x in data_dict[i]["img_path"]])
            img_path_list.extend(data_dict[i]["img_path"])
            gt_path_list.extend(data_dict[i]["gt_path"])
            caption_list.extend(data_dict[i]["caption"])


        self.dataset["data_name"] = dt_name_list
        self.dataset["img_path"] = img_path_list
        self.dataset["ori_img_path"] = deepcopy(img_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)

    def __len__(self):
        return len(self.dataset["img_path"])
    
    def __getitem__(self, idx):
        img_path = self.dataset["img_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        caption = self.dataset["caption"][idx]
        im = cv2.imread(img_path)
        gt = cv2.imread(gt_path)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)
        
        assert torch.max(gt) != 0, f"{gt_path} \n"
        
        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt,
            "caption": caption,
            "shape": torch.tensor(im.shape[-2:]),
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        image_tensor = torch.transpose(sample["image"], 0, 2).transpose(0, 1)
    
        image_tensor = image_tensor * 255 
        image_tensor = image_tensor.byte()   
        pil_image = Image.fromarray(image_tensor.numpy())

        sample["pil_image"] = pil_image
        sample["ori_label"] = gt.type(torch.uint8)  
        sample['ori_img_path'] = self.dataset["img_path"][idx]
        sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample