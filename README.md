# Training instruction for Our Project

We organize the training folder as follows.
```
train
|____data
|____pretrained_checkpoints
|____out
|____train_SAM_HQ.py
|____train_Net1.py
|____train_Net2.py
|____train_Net1_2.py
|____utils
| |____dataloader.py
| |____misc.py
| |____loss_mask.py
|____segment_anything_training
```

## 1. Data Preparation

(1). Kvasir-SEG can be downloaded from [simula link](https://datasets.simula.no/kvasir-seg/)

(2). Retina Blood Vessels datasets can be downloaded from [kaggle link](https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel)

(3). Brain MRI datasets can be downloaded from [Kaggle link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data)

(4). Breast Ultrasound Images datasets can be downloaded from [Kaggle link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

### Expected dataset structure for HQSeg-44K

```
data
|____Name of the Dataset
| |____train
| | |____images
| | |____masks
| |____val
| | |____images
| | |____masks
|

```
## 2. Init Checkpoint
Init checkpoint can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/pretrained_checkpoint)

### Expected checkpoint

```
pretrained_checkpoint
|____sam_vit_l_maskdecoder.pth
|____sam_vit_l_0b3195.pth

```

## 3. Training
To train any model on our datasets

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train_<model_name>.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

## 4. Evaluation
To evaluate any model on our datasets (and visualize the results)

```
python -m torch.distributed.launch --nproc_per_node=<num_gpus> train_<model_name>.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output> --eval --restore-model <path/to/training_checkpoint> --visualize
```
