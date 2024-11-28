import torch
from functools import partial

from model import ClipCellSam, ClipCellDecoder, ClipEncoder
from segment_anything import sam_model_registry

def build_model_biomedclip(
    sam_model_type: str,
    sam_checkpoint: str,
    model_checkpoint: str = None,
):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.eval()
    image_encoder = sam.image_encoder
    model = ClipCellSam(
        sam_image_encoder=image_encoder,
        clip_encoder=ClipEncoder(
            embedding_dim=512,
            out_dim=256,
            clip_model = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            tokenizer = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            learnable_pe = True,
        ),
        ClipCell_decoder=ClipCellDecoder(
            model_type=sam_model_type,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if model_checkpoint is not None:
        with open(model_checkpoint, "rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    return model

def build_model_laion_clip(
    sam_model_type: str,
    sam_checkpoint: str,
    model_checkpoint: str = None,
):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.eval()
    image_encoder = sam.image_encoder
    model = ClipCellSam(
        sam_image_encoder=image_encoder,
        clip_encoder=ClipEncoder(
            embedding_dim=1280,
            out_dim=256,
            context_length=77,
            clip_model = 'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
            tokenizer = 'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
            learnable_pe = True,
        ),
        ClipCell_decoder=ClipCellDecoder(
            model_type=sam_model_type,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if model_checkpoint is not None:
        with open(model_checkpoint, "rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    return model