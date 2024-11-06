import torch
from functools import partial

from model import FGCtrlClipSam, FGCtrlDecoder, ClipEncoder
from segment_anything import sam_model_registry

model_type_dict = {
    "4patches_256": {
        # "n_patches" : 4,
        "input_dim" : [256, 64],
        "output_dim" : [64, 32],
        "transformer_depth" : [2, 2],
        "downsample_rate": [2, 1]
    },
    "2patches_256": {
        # "n_patches" : 2,
        "input_dim" : [256, 64],
        "output_dim" : [64, 32],
        "transformer_depth" : [2, 2],
        "downsample_rate": [2, 1]
    },
    "4patches_512": {
        # "n_patches" : 4,
        "input_dim" : [256, 128, 64],
        "output_dim" : [128, 64, 32],
        "transformer_depth" : [2, 1, 1],
        "downsample_rate": [2, 2, 1]
    }
}

def build_model_biomedclip(
    sam_model_type: str,
    sam_checkpoint: str,
    model_type: str,
    model_checkpoint: str = None,
):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.eval()
    image_encoder = sam.image_encoder
    fgctrl_config = model_type_dict[model_type]
    model = FGCtrlClipSam(
        sam_image_encoder=image_encoder,
        clip_encoder=ClipEncoder(
            embedding_dim=512,
            out_dim=256,
            clip_model = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            tokenizer = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            learnable_pe = True,
            num_text_embeddings = 1
        ),
        fgctrl_decoder=FGCtrlDecoder(
            fgctrl_config=fgctrl_config,
            input_dim=256,
            output_dim=fgctrl_config["output_dim"][-1],
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
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
    model_type: str,
    model_checkpoint: str = None,
):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.eval()
    image_encoder = sam.image_encoder
    fgctrl_config = model_type_dict[model_type]
    model = FGCtrlClipSam(
        sam_image_encoder=image_encoder,
        clip_encoder=ClipEncoder(
            embedding_dim=1280,
            out_dim=256,
            context_length=77,
            clip_model = 'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
            tokenizer = 'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
            learnable_pe = True,
            num_text_embeddings = 1
        ),
        fgctrl_decoder=FGCtrlDecoder(
            fgctrl_config=fgctrl_config,
            input_dim=256,
            output_dim=fgctrl_config["output_dim"][-1],
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if model_checkpoint is not None:
        with open(model_checkpoint, "rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    return model